import numpy as np
import torch
import torch.nn as n
n
def logsumexp(x, dim=1):
    return torch.logsumexp(x.float(), dim=dim).type_as(x)

class DynamicCRF(nn.Module):
    def __init__(self, num_embedding, low_rank=32, beam_size=64):
        super().__init__()

        self.E1 = nn.Embedding(num_embedding, low_rank) #! 初始化两个embedding层
        self.E2 = nn.Embedding(num_embedding, low_rank)

        self.vocb = num_embedding #! 词表大小
        self.rank = low_rank #! 低秩
        self.beam = beam_size #! beam大小

    def extra_repr(self):
        return "vocab_size={}, low_rank={}, beam_size={}".format(
            self.vocb, self.rank, self.beam)

    def forward(self, emissions, targets, masks, beam=None):
        numerator = self._compute_score(emissions, targets, masks)
        denominator = self._compute_normalizer(emissions, targets, masks, beam)
        return numerator - denominator

    def forward_decoder(self, emissions, masks=None, beam=None):
        return self._viterbi_decode(emissions, masks, beam) #! crf decoding

    def _compute_score(self, emissions, targets, masks=None):
        batch_size, seq_len = targets.size() #! 获取batch大小和序列长度
        emission_scores = emissions.gather(2, targets[:, :, None])[:, :, 0]  # B x T #! 获取每个token的emission得分
        transition_scores = (self.E1(targets[:, :-1]) * self.E2(targets[:, 1:])).sum(2) #! 获取每个token的transition得分

        scores = emission_scores #! 初始化scores
        scores[:, 1:] += transition_scores #! 加上transition得分

        if masks is not None:
            scores = scores * masks.type_as(scores) #! 根据mask进行处理
        return scores.sum(-1) #! 返回每个序列的得分

    def _compute_normalizer(self, emissions, targets=None, masks=None, beam=None):
        beam = beam if beam is not None else self.beam #
        batch_size, seq_len = emissions.size()[:2] #! 获取batch大小和序列长度
        if targets is not None:
            _emissions = emissions.scatter(2, targets[:, :, None], np.float('inf')) #! 根据targets生成新的emissions
            beam_targets = _emissions.topk(beam, 2)[1] #! 获取beam_size个最大值的索引
            beam_emission_scores = emissions.gather(2, beam_targets) #! 获取beam_size个最大值的得分
        else:
            beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D; position i - 1, previous step.
        beam_transition_score2 = self.E2(beam_targets[:, 1:])   # B x (T-1) x K x D; position i, current step.
        beam_transition_matrix = torch.bmm(
            beam_transition_score1.view(-1, beam, self.rank),
            beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2))
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam)

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        for i in range(1, seq_len):
            next_score = score[:, :, None] + beam_transition_matrix[:, i-1]
            next_score = logsumexp(next_score, dim=1) + beam_emission_scores[:, i]

            if masks is not None:
                score = torch.where(masks[:, i:i+1], next_score, score)
            else:
                score = next_score

        # Sum (log-sum-exp) over all possible tags
        return logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, masks=None, beam=None):
        beam = beam if beam is not None else self.beam #! 获取beam大小
        batch_size, seq_len = emissions.size()[:2] #! 获取batch大小和序列长度
        beam_emission_scores, beam_targets = emissions.topk(beam, 2) #! 获取beam_size个最大值的得分和索引
        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D #! 获取每个token的transition得分
        beam_transition_score2 = self.E2(beam_targets[:, 1:])   # B x (T-1) x K x D #! 获取每个token的transition得分
        beam_transition_matrix = torch.bmm(
            beam_transition_score1.view(-1, beam, self.rank),
            beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2)) #! 计算transition矩阵
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam) #! 将transition矩阵展平

        traj_tokens, traj_scores = [], [] #! 初始化轨迹tokens和轨迹得分
        finalized_tokens, finalized_scores = [], [] #! 初始化最终tokens和最终得分

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        dummy = torch.arange(beam, device=score.device).expand(*score.size()).contiguous() #! 生成dummy

        for i in range(1, seq_len):
            traj_scores.append(score) #! 添加轨迹得分
            _score = score[:, :, None] + beam_transition_matrix[:, i-1] #! 计算得分
            _score, _index = _score.max(dim=1) #! 获取最大得分和索引
            _score = _score + beam_emission_scores[:, i] #! 加上emission得分

            if masks is not None:
                score = torch.where(masks[:, i: i+1], _score, score) #! 根据mask进行处理
                index = torch.where(masks[:, i: i+1], _index, dummy) #! 根据mask进行处理    
            else:
                score, index = _score, _index #! 更新得分和索引
            traj_tokens.append(index) #! 添加轨迹tokens

        # now running the back-tracing and find the best
        best_score, best_index = score.max(dim=1) #! 获取最大得分和索引
        finalized_tokens.append(best_index[:, None]) #! 添加最终tokens
        finalized_scores.append(best_score[:, None]) #! 添加最终得分

        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1] #! 获取最后一个token
            finalized_tokens.append(idx.gather(1, previous_index)) #! 添加轨迹tokens
            finalized_scores.append(scs.gather(1, previous_index)) #! 添加轨迹得分

        finalized_tokens.reverse() #! 反转最终tokens
        finalized_tokens = torch.cat(finalized_tokens, 1) #! 拼接最终tokens
        finalized_tokens = beam_targets.gather(2, finalized_tokens[:, :, None])[:, :, 0] #! 根据beam_targets获取最终tokens

        finalized_scores.reverse() #! 反转最终得分
        finalized_scores = torch.cat(finalized_scores, 1) #! 拼接最终得分
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1] #! 计算最终得分

        return finalized_scores, finalized_tokens
