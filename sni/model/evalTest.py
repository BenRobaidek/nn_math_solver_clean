import torch
from torch import autograd, nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data, datasets

def eval(data_iter, model, TEXT, emb_dim):
    model.eval()
    corrects, avg_loss, t5_corrects, rr = 0, 0, 0, 0
    for batch_count,batch in enumerate(data_iter):
        #print('avg_loss:', avg_loss)
        inp, target = batch.text, batch.label
        inp.data.t_()#, target.data.sub_(1)  # batch first, index align
        #inp3d = torch.cuda.FloatTensor(inp.size(0),inp.size(1),emb_dim)
        #for i in range(inp.size(0)):
        #  for j in range(inp.size(1)):
        #    inp3d[i,j,:] = vecs[TEXT.vocab.itos[inp[i,j].data[0]]]
        #if args.cuda:
        #    feature, target = feature.cuda(), target.cuda()

        logit = model(inp) #(Variable(inp))
        loss = F.cross_entropy(logit, target)#, size_average=False)
        ####
        avg_loss += loss.data[0]
        _, preds = torch.max(logit, 1)
        corrects += preds.data.eq(target.data).sum()
        # Rank 5
        """
        _, t5_indices = torch.topk(logit, 5)
        x = torch.unsqueeze(target.data, 1)
        target_index = torch.cat((x, x, x, x, x), 1)
        t5_corrects += t5_indices.data.eq(target_index).sum()
        """
        # Mean Reciprocal Rank
        _, rank = torch.sort(logit, descending=True)
        target_index = rank.data.eq(torch.unsqueeze(target.data, 1).expand(rank.size()))
        y = torch.arange(1, rank.size()[1]+1).view(1,-1).expand(rank.size())
        cuda = int(torch.cuda.is_available())-1
        if cuda == 0:
            y = y.cuda()
        y = (y.long() * target_index.long()).sum(1).float().reciprocal()
        rr += y.sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = 100.0 * corrects/size
    t5_acc = 100.0 * t5_corrects/size
    mrr = rr/size
    model.train()

    """
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) t5_acc: {:.4f}%({}/{}) MRR: {:.6f}\n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size,
                                                                       t5_acc,
                                                                       t5_corrects,
                                                                       size,
                                                                       mrr))
    """
    return(avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr);

def fast_test(inp, model):
    output = model(inp)
    _, predicted = torch.max(output, 1)
    return predicted

def test(text, model, text_field, label_field, path):
    #print(text)
    model.eval()
    fields = [('text', text_field), ('label', label_field)]
    example = data.Example.fromlist([text, ''], fields)
    dataset = data.Dataset([example], fields)
    train = data.TabularDataset(path=path, format='tsv', fields=fields)
    text_field.build_vocab(train)
    label_field.build_vocab(train)

    inp = None
    iterator = data.Iterator(dataset, batch_size=1)
    iterator.repeat=False
    for batch in iterator:
        inp = batch.text.t()
    output = model(inp)
    #print('OUTPUT:', output)
    _, predicted = torch.max(output, 1)
    #print('PREDICTED:', predicted)
    return predicted
