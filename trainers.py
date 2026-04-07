import collections
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss
from analyze_T import TMatrixAnalyzer
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr


class Trainer:
    def __init__(self, model, adv_model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        self.adv_model = adv_model
        self.online_similarity_model = args.online_similarity_model
        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)

        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True)
        )

        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.optim_adv = Adam(self.adv_model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

    def __refresh_training_dataset(self, item_embeddings):
        user_seq, _, _, _ = get_user_seqs(self.args.data_file)
        self.args.online_similarity_model.update_embedding_matrix(item_embeddings)
        train_dataset = RecWithContrastiveLearningDataset(
            self.args, user_seq,
            data_type='train', similarity_model_type='hybrid'
        )
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=self.args.batch_size
        )
        return train_dataloader

    def train(self, epoch):
        if epoch > self.args.augmentation_warm_up_epoches:
            print("refresh dataset with updated item embedding")
            self.train_dataloader = self.__refresh_training_dataset(
                self.model.item_embeddings
            )
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(-1).float()
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def _contrastive_from_embeds(self, mixed, input_ids_orig):
        out_orig = self.model.transformer_encoder(input_ids_orig)  # [B, L, D]
        out_aug = self.model.transformer_encoder_from_embeds(mixed, input_ids_orig)  # [B, L, D]
        flat_orig = out_orig.reshape(out_orig.shape[0], -1)
        flat_aug = out_aug.reshape(out_aug.shape[0], -1)
        return self.cf_criterion(flat_orig, flat_aug)


class ASTARTrainer(Trainer):

    def __init__(self, model, adv_model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(ASTARTrainer, self).__init__(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args
        )

        self.alpha = getattr(args, 'alpha', 1.0)
        self.beta = getattr(args, 'beta', 0.5)
        self.warmup_epochs = getattr(args, 'warmup_epochs', 20)
        self.max_grad_norm = getattr(args, 'max_grad_norm', 5.0)
        self.item_similarity = getattr(args, 'item_similarity', None)

        if self.cuda_condition:
            self.adv_model.cuda()

        self.analyzer = TMatrixAnalyzer(args, N_rand=args.N_rand, N_sim=args.N_sim)

    def _phase1_recommender(self, input_ids, target_pos, target_neg, epoch):
        self.model.train()
        self.adv_model.eval()
        self.optim.zero_grad()

        # Original sequence
        seq_out_orig = self.model.transformer_encoder(input_ids)
        L_rec_orig = self.cross_entropy(seq_out_orig, target_pos, target_neg)
        L_rec_org = self.cross_entropy(
            seq_out_orig[:, -1:, :],
            target_pos[:, -1:],
            target_neg[:, -1:]
        )

        # Hard augmented view (augmenter frozen)
        with torch.no_grad():
            soft_mixed, hard_mixed, lam, T, own_mask = self.adv_model(
                input_ids,
                self.model.item_embeddings,
                item_similarity=self.item_similarity,
            )
        lam_mean = lam.mean().detach()

        # FIX: use hard_mixed in Phase-1 contrastive
        L_contrast = self._contrastive_from_embeds(soft_mixed, input_ids)

        L_B = (self.args.rec_weight * L_rec_orig
               + self.args.reclp_weight * L_rec_org
               + self.args.cf_weight  * L_contrast ) # * lam_mean

        L_B.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()

        return {
            'L_rec_orig': L_rec_orig.item(),
            'L_rec_org': L_rec_org.item(),
            'L_contrast': L_contrast.item(),
            'L_B': L_B.item(),
            'lam_mean': lam_mean.item(),
        }

    def _phase2_augmenter(self, input_ids, target_pos, target_neg):
        self.model.eval()
        self.adv_model.train()
        self.optim_adv.zero_grad()

        # Freeze recommender params
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Frozen original representation for contrastive anchor
        with torch.no_grad():
            seq_out_orig = self.model.transformer_encoder(input_ids)

        # Soft augmented view
        soft_mixed, hard_mixed, lam, T, own_mask = self.adv_model(
            input_ids,
            self.model.item_embeddings,
            item_similarity=self.item_similarity,
        )

        seq_out_aug = self.model.transformer_encoder_from_embeds(
            soft_mixed, input_ids
        )

        L_rec_aug = self.cross_entropy(
            seq_out_aug[:, -1:, :],
            target_pos[:, -1:],
            target_neg[:, -1:]
        )

        flat_orig = seq_out_orig.reshape(seq_out_orig.shape[0], -1)
        flat_aug = seq_out_aug.reshape(seq_out_aug.shape[0], -1)
        L_contrast = self.cf_criterion(flat_orig, flat_aug)

        L_A = self.beta * L_rec_aug - self.alpha * L_contrast

        L_A.backward()
        torch.nn.utils.clip_grad_norm_(self.adv_model.parameters(), self.max_grad_norm)
        self.optim_adv.step()

        # Unfreeze recommender
        for param in self.model.parameters():
            param.requires_grad_(True)

        return {
            'L_rec_aug_A': L_rec_aug.item(),
            'L_contrast_A': L_contrast.item(),
            'L_A': L_A.item(),
        }

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"

        if train:
            self.model.train()
            self.adv_model.train()

            metrics = {
                'L_rec_orig': 0.0,
                'L_rec_org': 0.0,
                'L_contrast': 0.0,
                'L_B': 0.0,
                'lam_mean': 0.0,
                'L_rec_aug_A': 0.0,
                'L_contrast_A': 0.0,
                'L_A': 0.0,
            }

            is_warmup = epoch < self.warmup_epochs
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                if is_warmup:
                    self.model.train()
                    self.optim.zero_grad()
                    seq_out = self.model.transformer_encoder(input_ids)
                    L_rec = self.cross_entropy(seq_out, target_pos, target_neg)
                    L_rec.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optim.step()
                    metrics['L_rec_orig'] += L_rec.item()
                    metrics['L_B'] += L_rec.item()

                else:
                    stats1 = self._phase1_recommender(input_ids, target_pos, target_neg, epoch)
                    stats2 = self._phase2_augmenter(input_ids, target_pos, target_neg)
                    for k, v in {**stats1, **stats2}.items():
                        if k in metrics:
                            metrics[k] += v

                if i % 10 == 0:
                    rec_cf_data_iter.set_description(
                        f"Epoch {epoch} "
                        f"{'[warmup]' if is_warmup else '[adv]'}: "
                        f"L_B={metrics['L_B']/(i+1):.4f} "
                        f"L_A={metrics['L_A']/(i+1):.4f} "
                        f"λ={metrics['lam_mean']/(i+1):.3f}"
                    )

            if not is_warmup:
                self.adv_model.decay_tau()

            num_batches = len(dataloader)
            for k in metrics:
                metrics[k] /= num_batches

            post_fix = {
                "epoch": epoch,
                "warmup": is_warmup,
                "L_rec_orig": f"{metrics['L_rec_orig']:.4f}",
                "L_rec_org": f"{metrics['L_rec_org']:.4f}",
                "L_contrast": f"{metrics['L_contrast']:.4f}",
                "L_B": f"{metrics['L_B']:.4f}",
                "lam_mean": f"{metrics['lam_mean']:.4f}",
                "L_rec_aug_A": f"{metrics['L_rec_aug_A']:.4f}",
                "L_contrast_A": f"{metrics['L_contrast_A']:.4f}",
                "L_A": f"{metrics['L_A']:.4f}",
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(
                enumerate(dataloader),
                desc="Recommendation EP_%s:%d" % (str_code, epoch),
                total=len(dataloader),
                bar_format="{l_bar}{r_bar}"
            )
            self.model.eval()
            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]
                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)


class CoSeRecTrainer(Trainer):

    def __init__(self, model, adv_model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(CoSeRecTrainer, self).__init__(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args
        )

    def _one_pair_contrastive_learning(self, inputs):
        cl_batch = torch.cat(inputs, dim=0).to(self.device)
        cl_sequence_output = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        return self.cf_criterion(cl_output_slice[0], cl_output_slice[1])

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for _ in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                sequence_output = self.model.transformer_encoder(input_ids)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)

                joint_loss = self.args.rec_weight * rec_loss
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                for j, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[j] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss": '{:.4f}'.format(
                    cl_sum_avg_loss / (len(rec_cf_data_iter) * self.total_augmentaion_pairs)
                ),
            }
            for j, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix[f'cl_pair_{j}_loss'] = '{:.4f}'.format(
                    cl_individual_avg_loss / len(rec_cf_data_iter)
                )

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(
                enumerate(dataloader),
                desc="Recommendation EP_%s:%d" % (str_code, epoch),
                total=len(dataloader),
                bar_format="{l_bar}{r_bar}"
            )
            self.model.eval()
            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]
                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)

# ─────────────────────────────────────────────────────────────────────────────
# ASTARDiversityTrainer
# ─────────────────────────────────────────────────────────────────────────────

class ASTARDiversityTrainer(Trainer):
    """Diversity-and-exploration augmenter trainer (ablation: ASTARDiversity).

    Augmenter loss:
        L_A = β  * L_rec_aug                  (info preservation)
            + diversity_w * L_strategy_diversity  (−EMD vs past strategies)
            + view_div_w  * L_view_divergence     (−JSD between T1 and T2)
            − η           * L_entropy(T1, T2)     (encourage non-degenerate T)

    Recommender loss:
        L_B = rec_weight * L_rec_orig + cf_weight * L_CL(view1, view2)
    """

    def __init__(self, model, adv_model,
                 train_dataloader, eval_dataloader, test_dataloader, args):
        super().__init__(
            model, adv_model,
            train_dataloader, eval_dataloader, test_dataloader,
            args
        )

        self.beta         = getattr(args, 'astar_beta',        50.0)
        self.warmup_epochs = getattr(args, 'warmup_epochs',    20)
        self.max_grad_norm = getattr(args, 'max_grad_norm',    5.0)
        self.item_similarity = getattr(args, 'item_similarity', None)

        self.diversity_w  = getattr(args, 'diversity_weight',  0.1)
        self.view_div_w   = getattr(args, 'view_div_weight',   0.1)
        self.entropy_w    = getattr(args, 'entropy_weight',    0.01)
        self.window_size  = getattr(args, 'strategy_window_size', 10)

        # Sliding window of past strategy fingerprints (numpy arrays)
        self._strategy_window = collections.deque(maxlen=self.window_size)
        # Accumulate batch fingerprints within the current epoch
        self._epoch_fingerprints: list = []

        if self.cuda_condition:
            self.adv_model.cuda()

        self.analyzer = TMatrixAnalyzer(args, N_rand=args.N_rand, N_sim=args.N_sim)

    # ── Pool-region helpers ───────────────────────────────────────────────────

    def _pool_boundaries(self):
        """Return (own_start, n_ops, op_names) for the current augmenter."""
        m = self.adv_model
        own_start = 1 + m.N_rand + m.N_sim + m.N_hist
        op_names  = ['mask', 'random_sub']
        if m.N_sim  > 0:
            op_names.append('sim_sub')
        if m.N_hist > 0:
            op_names.append('hist_sub')
        op_names += ['own_identity', 'own_shuffle']
        return own_start, op_names
    
    def _one_pair_contrastive_learning(self, inputs):
        cl_batch = torch.cat(inputs, dim=0).to(self.device)
        cl_sequence_output = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        return self.cf_criterion(cl_output_slice[0], cl_output_slice[1])

    # ── Soft operation proportions (differentiable) ───────────────────────────

    def _soft_op_dist(self, T: torch.Tensor, own_mask: torch.Tensor) -> torch.Tensor:
        """Return soft operation distribution [B, n_ops] from T [B, P, L].

        Each row sums to 1 and is differentiable w.r.t. T.
        """
        B, P, L    = T.shape
        device     = T.device
        m          = self.adv_model
        own_start, op_names = self._pool_boundaries()

        valid     = own_mask.float()              # [B, L]
        T_w       = T * valid.unsqueeze(1)        # [B, P, L]  zero at padding
        T_sum_pos = T_w.sum(dim=2)                # [B, P]  summed over output positions

        total_w = T_sum_pos.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]

        parts = []

        # mask
        parts.append(T_sum_pos[:, 0:1])

        # random_sub
        rand_end = 1 + m.N_rand
        parts.append(T_sum_pos[:, 1:rand_end].sum(dim=1, keepdim=True))

        # sim_sub
        if m.N_sim > 0:
            sim_start = rand_end
            sim_end   = sim_start + m.N_sim
            parts.append(T_sum_pos[:, sim_start:sim_end].sum(dim=1, keepdim=True))

        # hist_sub
        if m.N_hist > 0:
            hist_start = rand_end + m.N_sim
            hist_end   = hist_start + m.N_hist
            parts.append(T_sum_pos[:, hist_start:hist_end].sum(dim=1, keepdim=True))

        # own_identity vs own_shuffle
        T_own_pool = T_w[:, own_start:, :]        # [B, L_own, L]
        L_own = T_own_pool.shape[1]
        d     = min(L_own, L)
        diag_indices  = torch.arange(d, device=device)

        identity_w = T_own_pool[:, diag_indices, diag_indices].sum(dim=1, keepdim=True)  # [B, 1]
        own_total  = T_sum_pos[:, own_start:].sum(dim=1, keepdim=True)   # [B, 1]
        shuffle_w  = (own_total - identity_w).clamp(min=0.0)

        parts.extend([identity_w, shuffle_w])

        op_dist = torch.cat(parts, dim=1) / total_w   # [B, n_ops]
        return op_dist

    # ── Diversity losses ──────────────────────────────────────────────────────

    def _emd_loss(self, op_dist: torch.Tensor) -> torch.Tensor:
        """Wasserstein-1 (EMD) between current batch dist and window mean.

        Returns −EMD (we want to *maximise* distance → minimise negative).
        If the window is empty returns zero.
        """
        if len(self._strategy_window) == 0:
            return op_dist.new_zeros(1).squeeze()

        # Reference: mean of stored fingerprints (numpy → tensor)
        ref_np  = np.stack(self._strategy_window, axis=0).mean(0)  # [n_ops]
        ref     = torch.from_numpy(ref_np).to(dtype=op_dist.dtype, device=op_dist.device)

        # Current: mean over batch
        current = op_dist.mean(dim=0)          # [n_ops]

        # Wasserstein-1 = sum |CDF_p − CDF_q|
        diff_cdf = torch.abs(current.cumsum(0) - ref.cumsum(0))
        emd      = diff_cdf.sum()

        return -emd   # negative → minimise = maximise diversity

    def _jsd_loss(self, T1: torch.Tensor, T2: torch.Tensor,
                own_mask: torch.Tensor) -> torch.Tensor:
        """JSD between per-sample operation distributions of T1 and T2.

        Operates on the collapsed [B, n_ops] distribution (dense) rather than
        the raw [B, P, L] transport matrix (sparse, causes vanishing gradients).
        Returns -mean(JSD) — minimise to maximise divergence between heads.
        """
        eps = 1e-8

        d1 = self._soft_op_dist(T1, own_mask)   # [B, n_ops]
        d2 = self._soft_op_dist(T2, own_mask)   # [B, n_ops]
        m  = 0.5 * (d1 + d2)                    # [B, n_ops]

        kl1 = (d1 * (torch.log(d1 + eps) - torch.log(m + eps))).sum(dim=-1)  # [B]
        kl2 = (d2 * (torch.log(d2 + eps) - torch.log(m + eps))).sum(dim=-1)  # [B]
        jsd = 0.5 * (kl1 + kl2)                 # [B],  in [0, log2]

        return -jsd.mean()

    def _entropy_loss(self, T1: torch.Tensor, T2: torch.Tensor,
                      own_mask: torch.Tensor) -> torch.Tensor:
        """Mean column-wise entropy of T1 and T2 (both).

        Maximise entropy → minimise negative entropy.  Returned as positive
        value to be subtracted in the final loss.
        """
        eps    = 1e-8
        valid  = own_mask.float()
        n_valid = valid.sum().clamp(min=1.0)

        def col_entropy(T):
            H = -(T * torch.log(T + eps)).sum(dim=1)     # [B, L]
            return (H * valid).sum() / n_valid

        return (col_entropy(T1) + col_entropy(T2)) * 0.5

    # ── Fingerprint tracking ─────────────────────────────────────────────────

    @torch.no_grad()
    def _update_fingerprint_buffer(self, T1, T2, lam, own_mask):
        """Compute a batch fingerprint and store for epoch aggregation."""
        # Average T1 and T2 for the fingerprint
        T_avg   = (T1 + T2) * 0.5
        op_dist = self._soft_op_dist(T_avg, own_mask)    # [B, n_ops]
        fp      = op_dist.mean(dim=0).cpu().numpy()      # [n_ops]
        self._epoch_fingerprints.append(fp)

    def _end_of_epoch_update(self):
        """Aggregate epoch fingerprints and push to sliding window."""
        if self._epoch_fingerprints:
            epoch_fp = np.stack(self._epoch_fingerprints, axis=0).mean(0)  # [n_ops]
            self._strategy_window.append(epoch_fp)
            self._epoch_fingerprints = []

    # ── Training phases ───────────────────────────────────────────────────────

    def _phase1_recommender(self, input_ids, target_pos, target_neg, epoch, batch_idx, cl_batches):
        self.model.train()
        self.adv_model.eval()
        self.optim.zero_grad()

        seq_out_orig = self.model.transformer_encoder(input_ids)
        L_rec_orig   = self.cross_entropy(seq_out_orig, target_pos, target_neg)

        # Two hard views (augmenter frozen)
        with torch.no_grad():
            _, _, hard1, hard2, lam, T1, T2, own_mask = self.adv_model.forward_dual(
                input_ids, self.model.item_embeddings,
                item_similarity=self.item_similarity,
            )

        lam_mean = lam.mean().detach()

        # Record T matrix for visualisation (alternate T1 / T2)
        if epoch % 5 == 0:
            self.analyzer.record(T1, lam, own_mask, epoch, T2=T2)

        # View-to-view contrastive
        out1   = self.model.transformer_encoder_from_embeds(hard1, input_ids)
        out2   = self.model.transformer_encoder_from_embeds(hard2, input_ids)
        flat1  = out1.reshape(out1.shape[0], -1)
        flat2  = out2.reshape(out2.shape[0], -1)
        L_CL   = self.cf_criterion(flat1, flat2)

        cl_losses = []
        for cl_batch in cl_batches:
            cl_loss = self._one_pair_contrastive_learning(cl_batch)
            cl_losses.append(cl_loss)

        cl_loss_sum = sum(self.args.cf_weight * cl_loss for cl_loss in cl_losses)

        L_B = self.args.rec_weight * L_rec_orig + self.args.cf_weight * L_CL

        L_B.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()

        return {
            'L_rec_orig': L_rec_orig.item(),
            'L_CL':       L_CL.item(),
            'L_B':        L_B.item(),
            'cl_loss_sum': cl_loss_sum.item(),
            'lam_mean':   lam_mean.item(),
        }

    def _phase2_augmenter(self, input_ids, target_pos, target_neg):
        self.model.eval()
        self.adv_model.train()
        self.optim_adv.zero_grad()

        for param in self.model.parameters():
            param.requires_grad_(False)

        # Soft dual views
        soft1, soft2, _, _, lam, T1, T2, own_mask = self.adv_model.forward_dual(
            input_ids, self.model.item_embeddings,
            item_similarity=self.item_similarity,
        )

        # Info-preservation loss on view 1 and 2 (soft, gradients flow)
        seq_aug1 = self.model.transformer_encoder_from_embeds(soft1, input_ids)
        seq_aug2 = self.model.transformer_encoder_from_embeds(soft2, input_ids)

        L_rec_aug = self.cross_entropy(
            seq_aug1[:, -1:, :], target_pos[:, -1:], target_neg[:, -1:]
        ) + self.cross_entropy(
            seq_aug2[:, -1:, :], target_pos[:, -1:], target_neg[:, -1:]
        )

        # Soft operation distributions for diversity losses
        op_dist1 = self._soft_op_dist(T1, own_mask)   # [B, n_ops]
        op_dist2 = self._soft_op_dist(T2, own_mask)   # [B, n_ops]
        op_dist  = (op_dist1 + op_dist2) * 0.5         # average both heads

        L_diversity = self._emd_loss(op_dist)
        L_view_div  = self._jsd_loss(T1, T2, own_mask)
        L_entropy   = self._entropy_loss(T1, T2, own_mask)

        # FIX: flatten before passing to NCELoss
        flat_aug1  = seq_aug1.reshape(seq_aug1.shape[0], -1)
        flat_aug2  = seq_aug2.reshape(seq_aug2.shape[0], -1)
        L_contrast = self.cf_criterion(flat_aug1, flat_aug2)

        L_A = (self.beta       * L_rec_aug
               + self.diversity_w * L_diversity
               + self.view_div_w  * L_view_div
               - self.entropy_w   * L_entropy
            #    - 0.1  * L_contrast
               )

        L_A.backward()
        torch.nn.utils.clip_grad_norm_(self.adv_model.parameters(), self.max_grad_norm)
        self.optim_adv.step()

        # Accumulate fingerprint for end-of-epoch update (detached)
        self._update_fingerprint_buffer(
            T1.detach(), T2.detach(), lam.detach(), own_mask.detach()
        )

        for param in self.model.parameters():
            param.requires_grad_(True)

        return {
            'L_rec_aug':  L_rec_aug.item(),
            'L_diversity': L_diversity.item(),
            'L_view_div': L_view_div.item(),
            'L_entropy':  L_entropy.item(),
            'L_A':        L_A.item(),
            'L_contrast': L_contrast.item(),
        }

    # ── Main iteration ────────────────────────────────────────────────────────

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"

        if train:
            self.model.train()
            self.adv_model.train()

            metrics = {
                'L_rec_orig': 0.0, 'L_CL': 0.0, 'L_B': 0.0,
                'lam_mean': 0.0,
                'L_rec_aug': 0.0, 'L_diversity': 0.0,
                'L_view_div': 0.0, 'L_entropy': 0.0, 'L_A': 0.0,
            }

            is_warmup        = epoch < self.warmup_epochs
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                rec_batch  = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                if is_warmup:
                    self.model.train()
                    self.optim.zero_grad()
                    seq_out  = self.model.transformer_encoder(input_ids)
                    L_rec    = self.cross_entropy(seq_out, target_pos, target_neg)
                    L_rec.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optim.step()
                    metrics['L_rec_orig'] += L_rec.item()
                    metrics['L_B']        += L_rec.item()

                else:
                    stats1 = self._phase1_recommender(
                        input_ids, target_pos, target_neg, epoch, i, 
                        cl_batches
                    )
                    stats2 = self._phase2_augmenter(
                        input_ids, target_pos, target_neg
                    )
                    for k, v in {**stats1, **stats2}.items():
                        if k in metrics:
                            metrics[k] += v

                if i % 10 == 0:
                    rec_cf_data_iter.set_description(
                        f"Epoch {epoch} "
                        f"{'[warmup]' if is_warmup else '[div]'}: "
                        f"L_B={metrics['L_B']/(i+1):.4f} "
                        f"L_A={metrics['L_A']/(i+1):.4f} "
                        f"λ={metrics['lam_mean']/(i+1):.3f}"
                    )

            # Post-epoch: decay temperature and push fingerprint to window
            if not is_warmup:
                self.adv_model.decay_tau()
                self._end_of_epoch_update()

            num_batches = len(dataloader)
            for k in metrics:
                metrics[k] /= num_batches

            post_fix = {
                "epoch":       epoch,
                "warmup":      is_warmup,
                "L_rec_orig":  f"{metrics['L_rec_orig']:.4f}",
                "L_CL":        f"{metrics['L_CL']:.4f}",
                "L_B":         f"{metrics['L_B']:.4f}",
                "lam_mean":    f"{metrics['lam_mean']:.4f}",
                "L_rec_aug":   f"{metrics['L_rec_aug']:.4f}",
                "L_diversity": f"{metrics['L_diversity']:.4f}",
                "L_view_div":  f"{metrics['L_view_div']:.4f}",
                "L_entropy":   f"{metrics['L_entropy']:.4f}",
                "L_A":         f"{metrics['L_A']:.4f}",
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(
                enumerate(dataloader),
                desc="Recommendation EP_%s:%d" % (str_code, epoch),
                total=len(dataloader),
                bar_format="{l_bar}{r_bar}"
            )
            self.model.eval()
            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred      = self.predict_full(recommend_output)
                    rating_pred      = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind            = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind        = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[
                        np.arange(len(rating_pred))[:, None], arr_ind_argsort
                    ]
                    if i == 0:
                        pred_list   = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list   = np.append(pred_list,   batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items   = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]
                    test_logits      = self.predict_sample(recommend_output, test_neg_items)
                    test_logits      = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)
