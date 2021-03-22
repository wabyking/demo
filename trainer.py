import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SkipGramModel,TimestampedSkipGramModel
from data_reader import DataReader, Word2vecDataset,TimestampledWord2vecDataset
import json

import os
import argparse
import pickle
import numpy as np
# from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sys import platform
if platform != "darwin":
    plt.switch_backend('agg')


#coca 0 29  1990 - 2019
#coha 0 199  1810 2009
#arxiv 0 352 2007.4 - 2020.4
# nyt 1987- 2007
# nyt_yao 1986 - 2015

year_mapping = {
    # "coha.txt.raw.token.decade-output": ([(i-1810)//10 for i in range(1810, 2020, 10)],[str(i)+"s" for i in range(1810, 2020, 10)]),
    # "coca.txt.raw.token.decade-output": ([(i-1990)//10 for i in range(1990, 2020, 10)],[str(i)+"s" for i in range(1990, 2020, 10)]),
    # "coca.txt.raw.token-output": ([i-1990 for i in range(1990, 2020, 1)],[str(i) for i in range(1990, 2020, 1)]),
    # "coha.txt.raw.token-output": ([i-1810 for i in range(1810, 2009, 1)],[str(i) for i in range(1810, 2009, 1)]),
    # "arxiv.txt.raw.token-output": ([i for i in range(0, 352, 1)],["{}-{}".format( i//12 +1991, i%12+1 ) for i in range(0, 352, 1)]) ,
    # "nyt.txt.norm-output": ([i-1987 for i in range(1987, 2007, 1)],[str(i) for i in range(1987, 2007, 1)]),
    # "nyt_yao.txt-output": ([i-1986 for i in range(1986, 2015, 1)],[str(i) for i in range(1986, 2015, 1)]),
    "newsit.txt.norm-output": ([i-2007 for i in range(2007, 2019, 1)],[str(i) for i in range(2007, 2019, 1)]),
    "repubblica.txt.norm-output": ([i-1984 for i in range(1984, 2019, 1)],[str(i) for i in range(1984, 2019, 1)]),

}




#word_sin word_cos word_mixed word_linear word_mixed_fixed
parser = argparse.ArgumentParser(description='parameter information')
parser.add_argument('--time_type', dest='time_type', type=str,default= "word_mixed", help='sin cos  mixed others  linear, sin,  word_sin,word_cos,word_linear')
parser.add_argument('--text', dest='text', type=str,default= "coha.txt.train", help='text dataset')
parser.add_argument('--use_time', dest='use_time', default= 1, type=int, help='use_time or not')
parser.add_argument('--output', dest='output', default= "coha" , type=str, help='output dir to save embeddings')
parser.add_argument('--log_step', dest='log_step', default= 100 , type=int, help='log_step')
parser.add_argument('--from_scatch', dest='from_scatch', default= 1 , type=int, help='from_scatch or not')
parser.add_argument('--batch_size', dest='batch_size', default= 128, type=int, help='batch_size')
parser.add_argument('--emb_dimension', dest='emb_dimension', default= 50 , type=int, help='emb_dimension')
parser.add_argument('--add_phase_shift', dest='add_phase_shift', default= 0, type=int, help='add_phase_shift')
parser.add_argument('--verbose', dest='verbose', default= 0, type=int, help='verbose')
parser.add_argument('--lr', dest='lr', default= 0.01, type=float, help='learning rate')
parser.add_argument('--do_eval', dest='do_eval', default= 1, type=int, help='verbose')
parser.add_argument('--iterations', dest='iterations', default= 2, type=int, help='iterations')
parser.add_argument('--years', dest='years', default= 30, type=int, help='years')
parser.add_argument('--weight_decay', dest='weight_decay', default= 0, type=float, help='weight_decay')
parser.add_argument('--time_scale', dest='time_scale', default= 1, type=int, help='time_scale')
parser.add_argument('--min_count', dest='min_count', default= 25, type=int, help='min_count')
parser.add_argument('--window_size', dest='window_size', default= 5, type=int, help='window_size')

args = parser.parse_args()




if not  torch.cuda.is_available():
    args.verbose = 1



import numpy as np
import heapq
import scipy  

def keep_top(arr,k=3): 
    smallest = heapq.nlargest(k, arr)[-1]  # find the top 3 and use the smallest as cut off
    arr[arr < smallest] = 0 # replace anything lower than the cut off with 0
    return arr


def read_embeddings_from_file(file_name):
    embedding_dict = dict()
    with open(file_name,encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i==0:
                vocab_size,emb_dimension = [int(item) for item in line.split()]
                # embeddings= np.zeros([vocab_size,emb_dimension])
            else:
                tokens = line.split()
                word, vector = tokens[0], [float(num_str) for num_str in tokens[1:]]
                embedding_dict[word] = vector
    return embedding_dict




class Word2VecChecker:
    def __init__(self,path = "output",time_type = "word_sin"):
        # for time_type in os.listdir(path):
        #     if ".DS_Store" in time_type:
                # continue
        self.path = path
        subpath = os.path.join(path,time_type)
        if args.add_phase_shift:
            subpath += "_shift"
        if not os.path.exists(os.path.join(subpath,"vectors.txt")):
            print("cannot find vectors.txt in {}, try to find {}-th iteration".format(subpath,args.iterations))
            subpath = os.path.join(subpath,str(args.iterations-1))
            if not os.path.exists(subpath):
                print("cannot load model from {}".format(subpath))
                return
        self.embedding_dict = read_embeddings_from_file(os.path.join(subpath,"vectors.txt"))
        if args.use_time and  "word2vec" not in time_type:
            self.skip_gram_model = TimestampedSkipGramModel(len(self.embedding_dict), args.emb_dimension,time_type = time_type, add_phase_shift=args.add_phase_shift) 
        else:
            self.skip_gram_model = SkipGramModel(len(self.embedding_dict), args.emb_dimension)
        
        self.id2word = pickle.load(open(os.path.join(subpath, "dict.pkl"),"rb"))
        self.skip_gram_model.load_embeddings(self.id2word,subpath)




            # print(embeddings)
    def get_similar_words(self,words,year,k=3,word2id=None):
        if word2id is  None:
            word2id = {value:key for key,value in self.id2word.items()}
        embeddings_vectors = self.get_embedding_in_a_year(self.embedding_dict.keys(),word2id=word2id,year =year)
        
        # embeddings_vectors = np.array( [vector for word,vector in embeddings])
        # all_words = [word for word,vector in embeddings]
        not_found_words = [word for word in words if word not in word2id]
        if len(not_found_words) > 0:
            print("do not find {}".format(" ".join(not_found_words)) )
        words_index = [word2id[word] for word in words if word in word2id]
        # print(words_index)

        selected_vectors = np.array( [embeddings_vectors[word] for word in words_index])
        
        a = np.dot(selected_vectors,embeddings_vectors.T)#/np.norm()
        # a = cosine_similarity(selected_vectors,embeddings_vectors)
        
        top_k = a.argsort()[:,-1*k:]#[::-1]
        # top_k = np.partition(a, -3)
        # print(top_k.shape)
        # print(top_k)

        words_str  = [  " ".join([self.id2word[word]  for  word in top_k_per_word[::-1]])   for top_k_per_word in top_k ]
        return words_str

        # ranks = np.argsort(a,axis = 0)
        # print(ranks.argmax(0))
        # print(a.squeeze())
        # print(a.squeeze().argmax())
        # print(a.argmax(1))
        # print(a)
        # exit()
    def word_change_rate(self,words, years = 30):
        vectors = []
        for year in range(years):
            word2id = {value:key for key,value in self.id2word.items()}
            embeddings_vectors = self.get_embedding_in_a_year(self.embedding_dict.keys(),word2id=word2id,year =year)
            
            # embeddings_vectors = np.array( [vector for word,vector in embeddings])
            # all_words = [word for word,vector in embeddings]

            words_index = [word2id[word] for word in words]
            # print(words_index)

            selected_vectors = np.array( [embeddings_vectors[word] for word in words_index])
            vectors.append(selected_vectors)
        
        
        for j in range(len(words)):
            change_rates = []
            for year in range(years):
                if year ==0 :
                    cur_vector = vectors[year][j]
                else:
                    
                    # change_rate = np.dot(cur_vector,vectors[year][j])
                    change_rate = scipy.spatial.distance.cosine(cur_vector,vectors[year][j])
                    cur_vector = vectors[year][j]
                    change_rates. append(change_rate)
            print(words[j],np.mean(np.array(change_rates)))
            print(change_rates)
                

        return

    def plot_words_in_many_years(self,words= None, years = [i for i in range(1977,2020,1)],word2id=None,name="image"):
        if words is  None:
            words = ["president" , "reagan",  "trump", "biden", "obama","bush","carter","clinton", "ford", "nixon"]
            # words = ["weapon" , "nuclear",   "energy"]
        if word2id is None:
            word2id = {value:key for key,value in self.id2word.items()}
        vectors = []
        names = []
        for year in years:
            names.extend(["{}-{}".format(word,year) for word in words])
            embeddings = self.get_embedding_in_a_year(words,year,word2id)
            vectors.extend(embeddings)
        embed = TSNE(n_components=2).fit_transform(vectors)
        # print(embed.shape)

        plt.figure(figsize = (12,12))
        # from adjustText import adjust_text 
        texts = []
        for i,point in enumerate(embed):
            plt.scatter(point[0],point[1],label =names[i])
            texts.append(plt.text(point[0],point[1], names[i],size =7))
        # plt.plot(embed[:,0],embed[:,1],names)

        # adjust_text(texts)
        # plt.legend()
        if platform == "win32":
            plt.show()
        else:
            plt.savefig("president-{}.pdf".format(name),bbox_inches = "tight",pad_inches=0)
            plt.close()
        # plt.show()

    def get_sim_between_year(self,target,words= None,years = [i for i in range(1940,2020,1)],  word2id= None,name = "nuclear"):
        name += "-"+target+"_".join(words)
        sims = []
        words.append(target)
        
        for year in years:
            embeddings = self.get_embedding_in_a_year(words,year)
            sim = cosine_similarity(embeddings[-1][np.newaxis,:],embeddings[:-1]).squeeze()
            # print(sim.shape)
            sims.append(sim)
        sims = np.array(sims)
        plt.figure(figsize = (10,10))
        for i in range(len(sims[0])):
            plt.plot(years,sims[:,i],label = words[i])
        plt.legend(loc='upper left')
        if platform == "darwin_none":
            plt.show()
        else:
            plt.savefig("{}.pdf".format(name),bbox_inches = "tight",pad_inches=0)
            plt.close()
        


    def check_ssd(self,helper):

        from scipy.spatial.distance import cosine  # cosine distance

        words = helper.words
        time_stamped_embeddings = []
        for timespan in helper.timespans:
            all_embeddings = [self.get_embedding_in_a_year(words, year) for year in timespan ]
            mean_embedding = np.mean(np.array(all_embeddings),0)
            time_stamped_embeddings.append(mean_embedding)
        assert  len(time_stamped_embeddings) ==2 , "more timespans than two"
        scores = [cosine(time_stamped_embeddings[0][i],time_stamped_embeddings[1][i]) for i,word in enumerate(words)]
        print(scores)
        print(helper.evaluate(scores))









    def get_embedding_in_a_year(self,words= None, year = 0,word2id=None):
        if word2id is None:
            word2id = {value:key for key,value in self.id2word.items()}

        words_id = [word2id[word]for word in words]
        # print("___"*20)
        
        word,time = torch.LongTensor(words_id),torch.LongTensor([year]*len(words_id))
        # print(time)
        # print(word)
        embeddings = self.skip_gram_model.forward_embedding(word,time).data.numpy()
        return embeddings

def load_model(model,filename = "pytorch.bin"):

    state_dict = torch.load(filename)
    print(filename)
    print(state_dict.keys())
    print(state_dict.__class__.__name__)
    exit()
    missing_keys, unexpected_keys, error_msgs = [], [], []
    prefix = ""
    metadata = getattr(state_dict,"_metadata","None")
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix = ''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1],{})
        module._load_from_state_dict(state_dict, prefix,local_metadata,True,missing_keys,unexpected_keys,error_msgs)
        for name,child in module._modules.items():
            if child is not None:
                load(child,prefix + name + ".")
    start_prefix = ""
    load(model,prefix=start_prefix)

    if len(missing_keys) > 0:
        print("weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__,missing_keys))
    if len(unexpected_keys) > 0:
        print("weights of {} not used pretrained model: {}".format(model.__class__.__name__,unexpected_keys))
    if len(error_msgs) > 0:
        print("errors in loading state_dict  for  {}  :  \n{}".format(model.__class__.__name__,error_msgs))
    return model


class Word2VecTrainer:
    def __init__(self, args):# input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,initial_lr=0.01, min_count=25,weight_decay = 0, time_scale =1

        # self.data = DataReader(args.text, args.min_count)
        # if not args.use_time:
        #      dataset = Word2vecDataset(self.data, args.window_size)
        # else:
        #     dataset = TimestampledWord2vecDataset(self.data, args.window_size,args.time_scale)
        #
        # self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
        #                              shuffle=True, num_workers=0, collate_fn=dataset.collate)
        self.data,self.dataloader = self.load_train(args) # self.data

        if "train" in args.text:
            test_filename = args.text.replace("train","test")
            if  os.path.exists(test_filename):
                print("load test  dataset: ".format(test_filename))
                self.test = self.load_train(args, data = self.data, filename=test_filename, is_train=False )
            else:
                self.test = None

            dev_filename = args.text.replace("train", "dev")
            if  os.path.exists(dev_filename):
                print("load dev dataset: ".format(dev_filename))
                self.dev = self.load_train(args, data = self.data, filename=dev_filename, is_train=False)
            else:
                self.dev = None
        else:
            self.dev, self.test = None, None

        
        if args.use_time:
            self.output_file_name = "{}/{}".format(args.output, args.time_type)
            if args.add_phase_shift:
                self.output_file_name  += "_shift"
        else:
            self.output_file_name = "{}/{}".format(args.output, "word2vec")
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        if not os.path.exists(self.output_file_name):
            os.mkdir(self.output_file_name)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.emb_dimension
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.lr = args.lr
        self.time_type = args.time_type
        self.weight_decay = args.weight_decay

        print(args)


        if args.use_time:
            self.skip_gram_model = TimestampedSkipGramModel(self.emb_size, self.emb_dimension,time_type = args.time_type,add_phase_shift=args.add_phase_shift) 
        else:
            self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            print("using cuda and GPU ....")
            self.skip_gram_model.cuda()

        # load_path = "{}/{}".format(self.output_file_name)
        # torch.save(self.skip_gram_model,"pytorch.bin")
        # self.skip_gram_model =  torch.load("pytorch.bin")
        # self.skip_gram_model = load_model(self.skip_gram_model,"pytorch.bin")
        # exit()
        if not args.from_scatch and os.path.exists(self.output_file_name):

            print("loading parameters  ....")
            self.skip_gram_model.load_embeddings(self.data.id2word,self.output_file_name)

    def load_train(self,args,data= None, filename = None, is_train = True):
        if data is None:
            assert is_train==True, "wrong to load data 1"
            data = DataReader(args.text, args.min_count)
            filename = args.text
        else:
            assert is_train == False, "wrong to load test data 2"
            assert filename is not None, "wrong to load test data 3"
            assert data is not None, "wrong to load test data 4"
        if not args.use_time:
            dataset = Word2vecDataset(data, input_text = filename, window_size= args.window_size)
        else:
            dataset = TimestampledWord2vecDataset(data,input_text = filename, window_size= args.window_size, time_scale=args.time_scale)

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=is_train, num_workers=0, collate_fn=dataset.collate) # shuffle if it is train
        if is_train:
            return data,dataloader
        else:
            return dataloader

    def evaluation_loss(self,logger =None):
        results = []
        self.skip_gram_model.eval()
        print("evaluating ...")
        for index,dataloader in enumerate([self.dev,self.test]):
            if dataloader is None:
                continue
            losses = []
            for i, sample_batched in enumerate(tqdm(dataloader)):
                if len(sample_batched[0]) > 1:

                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    if args.use_time:
                        time = sample_batched[3].to(self.device)
                        # print(time)
                        loss, pos, neg = self.skip_gram_model.forward(pos_u, pos_v, neg_v, time)
                    else:

                        loss, pos, neg = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    # print(loss)
                    losses.append(loss.item())
            mean_result = np.array(losses).mean()
            results.append(mean_result)
            print("test{} loss is {}".format(index, mean_result))
            logger.write("Loss in  test{}: {} \n".format( index, str(mean_result)))
            logger.flush()

        self.skip_gram_model.train()
        return results

    def train(self):
        print(os.path.join(self.output_file_name,"log.txt"))
        if not os.path.exists(self.output_file_name):
            os.mkdir(self.output_file_name)
        optimizer = optim.Adam(self.skip_gram_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader)*self.iterations)


        with open("{}/log.txt".format(self.output_file_name,"log.txt"),"w") as f:
            for iteration in range(self.iterations):

                print("\nIteration: " + str(iteration + 1))
                f.write(str(args) +"\n")
                # optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)


                running_loss = 0.0
                for i, sample_batched in enumerate(tqdm(self.dataloader)):
                    if len(sample_batched[0]) > 1:

                        pos_u = sample_batched[0].to(self.device)
                        pos_v = sample_batched[1].to(self.device)
                        neg_v = sample_batched[2].to(self.device)

                        optimizer.zero_grad()
                        if args.use_time:
                            time = sample_batched[3].to(self.device)
                            # print(time)
                            loss,pos,neg = self.skip_gram_model.forward(pos_u, pos_v, neg_v,time)
                        else:

                            loss,pos,neg = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                        # print(loss)

                        loss.backward()
                        optimizer.step()
                        scheduler.step()



                        loss,pos,neg = loss.item(),pos.item(),neg.item()

                        if  i % args.log_step == 0: # i > 0 and
                            f.write("Loss in {} steps: {} {}, {}\n".format(i,str(loss),str(pos),str(neg)))

                        if  not torch.cuda.is_available() or i % (args.log_step*10) == 0 :
                            print("Loss in {} steps: {} {}, {}\n".format(i,str(loss),str(pos),str(neg)))
                self.evaluation_loss(logger=f)
                epoch_path = os.path.join(self.output_file_name,str(iteration))
                if not os.path.exists(epoch_path):
                    os.mkdir(epoch_path)

                torch.save(self.skip_gram_model, os.path.join( epoch_path,"pytorch.bin") )

                self.skip_gram_model.save_embedding(self.data.id2word, os.path.join(self.output_file_name,str(iteration)))
                self.skip_gram_model.save_in_text_format(self.data.id2word,
                                                         os.path.join(self.output_file_name, str(iteration)))
            self.skip_gram_model.save_in_text_format(self.data.id2word,self.output_file_name)


            torch.save(self.skip_gram_model, os.path.join(self.output_file_name,"pytorch.bin") )
            with open(os.path.join(self.output_file_name,"config.json"), "wt") as f:
                json.dump(vars(args), f, indent=4)
            self.skip_gram_model.save_dict(self.data.id2word,self.output_file_name)



def get_sim_words(checker, words, years,real_years, k = 100 ):
    simwords = []
    for year in years:
        simwords.append(checker.get_similar_words(words = words, year = year, k = k))

    # base_year = 1810 if "coha" in checker.path else 1990
    # real_years = [str(year + base_year) for year in  years]
    #
    # if "arxiv" in checker.path:
    #     real_years = ["{}-{}".format( (year-4)//12 +2007, (year-4)%12 ) for year in years]

    lines = ["{} ".format(checker.path)]
    for row in range(len(simwords[0])):
        line = [real_years[i] + " : " + simword[row] for i,simword in enumerate(simwords)]
        print(line)
        print("--"*20)
        lines.extend(line)
    return "\n".join(lines)


check_list = [ ("president", [    "nixon","ford","carter", "reagan","clinton", "bush" , "obama", "trump", "biden"]),
    ("olympic", [ "moscow",  "los", "angeles", "seoul", "barcelona","atlanta","sydney","athens", "beijing", "london", "rio", "tokyo"]),
    ("nuclear", [    "technology","threaten","america", "russian","cuba", "green" , "energy","china"]),
    ("nuclear", [    "russian","japan", "weapon" , "energy", "ukrainian", "soviet"]),
    ("olympic", ["sydney","athens", "beijing", "london", "rio", "tokyo"]),
    ("president", [ "clinton", "bush" , "obama", "trump", "biden"]),
]



coha_words = ["apple", "amazon" , "dna",  "innovation" , "data" , "app", "twitter",  "ranking","quantum", "nuclear","weapon", "president" , "chairman" ,"soviet", "reagan",  "trump", "biden", "obama", "olympic", "olympics", "china","america","ai", "artificial", "intelligence", "neural", "network", "language", "model","information", "retrieval"]
words = coha_words + ["iphone", "mp3"]

def draw_figure():
    for output in ["coha.txt.raw.token-output/", "coca.txt.raw.token-output/", "arxiv.txt.raw.token-output/"]:
        if "coca" in output:
            years = [i-1990 for i in range(1990, 2020, 1)]
        else:
            years = [i-1810 for i in range(1810, 2020, 1)]
        for time_type in ["word_mixed_fixed", "word_cos"]: # "word_cos",
            for epoch in range(1,10,1):
                args.iterations = epoch
                try:
                    checker = Word2VecChecker(path=output, time_type=time_type)
                    for target, checked_words in check_list:
                        # checker.plot_words_in_many_years(words=[target] + checked_words[-9:], years=years,
                        #                                  name="{}-{}".format(output.split(".")[0], time_type))
                        checker.get_sim_between_year(target, checked_words[-9:],
                                                     name="{}-{}-{}-".format(output.split(".")[0], time_type,epoch), years=years)
                except Exception as e:
                    print(e)


timetypes = ["cos" ,       "linear_shift", " mixed_shift",   "sin_shift",  "word_cos",        "word_linear_shift",  "word_mixed_fixed",        "word_mixed_shift",  "word_sin_shift",
"cos_shift  mixed",         "others_shift",  "word2vec",   "word_cos_shift",  "word_mixed",         "word_mixed_fixed_shift",  "word_sin"]


def check_ssd():
    from data.ssd import Helper

    helper = Helper("data/grade.txt")
    for time_type in timetypes:  # [ "word_sin" ,"word_cos", "word_cos_shift", "word_cos_shift" ,"word_mixed_fixed","cos","cos_shift",""]: #
        for epoch in range(10):
            try:
                print(time_type, epoch, "-" * 20 + "\n")
                args.iterations = epoch
                checker = Word2VecChecker(path="coha.txt.raw.token-output/", time_type=time_type)
                checker.check_ssd(helper)
            except Exception as e:
                print(e)

def sim_words_over_time(model_path,words,epoches = 10,dataset="none",years =()):

    years, real_years = years

    for time_type in ["word_mixed_fixed"]: # "word_cos", , "word_cos"
        epoches = 10 if "mixed_fixed" in time_type else 5

        for epoch in range(1,epoches,1):
            save_filename = "{}-{}-{}-sim_word_log.txt".format(dataset, epoch, time_type)
            print("save log in {}".format(save_filename))
            with open(save_filename, "w", encoding="utf-8") as f:
                args.iterations = epoch
                checker = Word2VecChecker(path=model_path, time_type=time_type)
                log_text = get_sim_words(checker, words, years,real_years)
                print(log_text)

                f.write(log_text + "\n")
                # exit()



words = ["dna", "innovazione", "invecchiamento", "anziano", "vaccino", "spaziale", "coronavirus", "pandemia","mascherina", "vaccino", "test", "respiratore"]


if __name__ == '__main__':
    
    if args.do_eval:
        # draw_figure()
        for model_path,(years, real_years) in year_mapping.items():
            sim_words_over_time(model_path,words, dataset=model_path.split("-")[0], years=(years, real_years))
            # if "coha" in model_path:
            #     sim_words_over_time(model_path,coha_words,dataset = model_path.split("-")[0], years =(years, real_years) )
            # else:
            #     sim_words_over_time(model_path,words,dataset = model_path.split("-")[0],years =(years, real_years))
             # checker.word_change_rate(words, years =args.years)
    else:
        w2v = Word2VecTrainer(args)
        #input_file = args.text, output_file = args.output, batch_size = args.batch_size, initial_lr = args.lr, weight_decay = args.weight_decay, iterations = args.iterations, time_scale = args.time_scale
        w2v.train()

    # embeddings = checker.get_embedding_in_a_year(words = "network", year =1990)
