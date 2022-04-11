    def forward(self, x):

        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1)

        word_rep = self.word_rep_layer(x)['h']

        span_rep = self.span_rep_layer(word_rep, span_idx)

        logits = self.classifier(span_rep)

        output = {'logits': logits}

        if self.training:

            y = x['span_label'].view(-1)

            y_pred = logits.view(-1, self.num_classes)

#             loss = F.cross_entropy(y_pred, y, reduction='sum', ignore_index=-1)
            loss = lab_smooth(y_pred, y, eps=0.45)

            output['loss'] = loss

        return output
    
import torch.nn.functional as F


def curriculum_loss(logits, y, tau=0.1):
    pred = logits.argmax(-1)
    mask_loss = ((y == 0) * (pred != 0)).float()
    weight = tau * mask_loss + (1 - mask_loss)
    loss = F.cross_entropy(logits, y, reduction='none', ignore_index=-1)
    return (loss * weight).sum()


def neg_sampling_loss(logits, y, sample_rate=0.9):
    rate = 1 - sample_rate
    sample = (torch.rand(size=y.size()) > rate).type_as(y)
    mask_neg = (sample == 1) * (y == 0)
    masked_y = y.masked_fill(mask_neg, value=-1)
    loss = F.cross_entropy(logits, masked_y, reduction='sum', ignore_index=-1)
    return loss

def equiloss(logits, y):
    
    loss_entity = F.cross_entropy(logits, y.masked_fill(y == 0, -1), ignore_index=-1, reduction='sum')
    loss_non_entity = F.cross_entropy(logits, y.masked_fill(y > 0, -1), ignore_index=-1, reduction='sum')
    
    return loss_entity + loss_non_entity * 0.15

def lab_smooth(logits, y, eps=0.2):
    
    num_classes = logits.shape[-1]
    
    loss_entity = F.cross_entropy(logits, y.masked_fill(y == 0, -1), ignore_index=-1, reduction='sum')
        
    y_0 = y.masked_select(y == 0)
    lp_0 = logits.masked_select((y == 0).unsqueeze(1)).view(-1, num_classes).log_softmax(dim=-1)
    one_hot_y0 = F.one_hot(y_0, num_classes)
    y_0 = (1 - eps) * one_hot_y0 + eps/(num_classes - 1) * (1 - one_hot_y0)
        
    loss_ne = -(lp_0 * y_0).sum()
    
    return loss_entity + loss_ne

def vanilla_loss(logits, y):
    loss = F.cross_entropy(logits, y, reduction='sum', ignore_index=-1)
    return loss

def confidence_aware(logits, y, tau=0.5):
    prob = torch.softmax(logits, -1)
    prob_zero = prob[:, 0]
    mask = (prob_zero < tau) * (y == 0)
    masked_y = y.masked_fill(mask, value=-1)
    log_prob = torch.log(prob)
    loss = F.nll_loss(log_prob, masked_y, reduction='sum', ignore_index=-1)
    return loss
