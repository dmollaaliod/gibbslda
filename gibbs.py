"""Simple implementation of Gibbs sampling.
By Diego Molla (dmollaaliod@gmail.com)
with contribution from Benjamin Boerschinger.
Based on the algorithm described in:
  Steyvers & Griffiths. Probabilistic Topic Models. In Landauer, McNamara, and
  Kintsch (eds), "Latent Semantic Analysis: A Road to Meaning". Laurence Erlbaum.
"""

import random

def generate_docs(words, phi, docs):
    "Generate documents randomly"
    dataset = []
    for d in docs:
        doc = []
        nwords = 16 # random.randint(50,100)
        for i in range(0,nwords):
            if random.random() < d[0]:
                topic = 0
            else:
                topic = 1
            doc.append(sample(phi[topic]))	#sample from the topic
        dataset.append(doc)
    return dataset

def initialise(data):
    """Return an initial distribution of topics
    >>> data = [['bank','money','money','bank','loan'],
    ...   ['stream','loan','river','loan','money'],
    ...   ['money','money','river','bank','stream','bank'],
    ...   ['bank','stream','bank','bank','river']]
    >>> (topic_distr,first_sample) = initialise(data)
    >>> topic_distr
    [[['bank', {0: 0.5, 1: 0.5}], ['money', {0: 0.5, 1: 0.5}], ['money', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}], ['loan', {0: 0.5, 1: 0.5}]], [['stream', {0: 0.5, 1: 0.5}], ['loan', {0: 0.5, 1: 0.5}], ['river', {0: 0.5, 1: 0.5}], ['loan', {0: 0.5, 1: 0.5}], ['money', {0: 0.5, 1: 0.5}]], [['money', {0: 0.5, 1: 0.5}], ['money', {0: 0.5, 1: 0.5}], ['river', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}], ['stream', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}]], [['bank', {0: 0.5, 1: 0.5}], ['stream', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}], ['river', {0: 0.5, 1: 0.5}]]]
"""
    topic_distr = []
    first_sample = []
    for d in data:
        d_topic_distr = []
        doc_sample = []
        for w in d:
            distr = {0:0.5,1:0.5}
            d_topic_distr.append([w,distr])
            doc_sample.append([w,sample(distr)])
        topic_distr.append(d_topic_distr)
        first_sample.append(doc_sample)
    return topic_distr,first_sample

def gibbs_iterate(this_sample,alpha=50/2,beta=0.01):
    """Return the result of an iteration of Gibbs sampling:
    % >>> doc_topics = [[['bank', {0: 0.5, 1: 0.5}], ['money', {0: 0.5, 1: 0.5}], ['money', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}], ['loan', {0: 0.5, 1: 0.5}]], [['stream', {0: 0.5, 1: 0.5}], ['loan', {0: 0.5, 1: 0.5}], ['river', {0: 0.5, 1: 0.5}], ['loan', {0: 0.5, 1: 0.5}], ['money', {0: 0.5, 1: 0.5}]], [['money', {0: 0.5, 1: 0.5}], ['money', {0: 0.5, 1: 0.5}], ['river', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}], ['stream', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}]], [['bank', {0: 0.5, 1: 0.5}], ['stream', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}], ['bank', {0: 0.5, 1: 0.5}], ['river', {0: 0.5, 1: 0.5}]]]
    % >>> gibbs_iterate(doc_topics)
    """

    # Generate next posterior estimate
    d_i = -1
    topic_distr = []
    for d in this_sample:
        d_topic_distr = []
        d_i += 1
        for w in d:
            CDT = compute_CDT(this_sample)
            CWT = compute_CWT(this_sample)

            # Compute the posterior distribution
            p_z = {}
            for t in range(2):
                p_z[t] = compute_pz(w,d_i,t,CWT,CDT,alpha,beta,this_sample)

            # Renormalize the posterior distribution
            Z = p_z[0]+p_z[1]
            p_z[0] = p_z[0]/Z
            p_z[1] = p_z[1]/Z

	    # Sample from the posterior
            w[1] = sample(p_z)

            d_topic_distr.append([w[0],p_z])

        topic_distr.append(d_topic_distr)

    return topic_distr, this_sample 

def compute_pz(w,d_i,t,CWT,CDT,alpha,beta,this_sample):
    """Return the estimated probability of assigning the word token to the topic,
    conditioned on the topic assignments to all other word tokens:
                                    CWT[w_i][j] + beta
    P(z_i=j|z_{-i},w_i,d_i,.) =..= ---------------------------------  *
                                    SUM_{w=1}^W(CWT[w][j])+W*beta
                                    
                                    CDT[d_i][j] + alpha
                                   ---------------------------------
                                    SUM_{t=1}^T(CDT[d_i][t])+T*alpha
                                    
    >>> this_sample = [[['bank', 1], ['money', 0], ['money', 1], ['bank', 1], ['loan', 0]], 
    ...   [['stream', 1], ['loan', 0], ['river', 1], ['loan', 0], ['money', 1]],
    ...   [['money', 1], ['river', 0], ['bank', 0], ['stream', 1], ['bank', 0]],
    ...   [['bank', 1], ['stream', 1], ['bank', 0], ['bank', 1], ['river', 1]]]
    >>> CDT = compute_CDT(this_sample)
    >>> CWT = compute_CWT(this_sample)
    >>> pz = compute_pz(('stream',1),1,1,CWT,CDT,50/2,0.01,this_sample)
    >>> pz == (2+0.01)*(2+50/2)/((11+5*0.01)*(4+50))
    True
    >>> pz = compute_pz(('stream',1),1,0,CWT,CDT,50/2,0.01,this_sample)
    >>> pz == (0+0.01)*(2+50/2)/((8+5*0.01)*(4+50))
    True
    >>> pz = compute_pz(('loan',0),1,1,CWT,CDT,50/2,0.01,this_sample)
    >>> pz == (0+0.01)*(3+50/2)/((12+5*0.01)*(4+50))
    True
    >>> pz = compute_pz(('loan',0),1,0,CWT,CDT,50/2,0.01,this_sample)
    >>> pz  == (2+0.01)*(1+50/2)/((7+5*0.01)*(4+50))
    True
    >>> pz = compute_pz(('river',1),1,1,CWT,CDT,50/2,0.01,this_sample)
    >>> pz == (1+0.01)*(2+50/2)/((11+5*0.01)*(4+50))
    True
    >>> pz = compute_pz(('river',1),1,0,CWT,CDT,50/2,0.01,this_sample)
    >>> pz == (1+0.01)*(2+50/2)/((8+5*0.01)*(4+50))
    True
    """
    W = len(CWT)
    T = 2

    cwt = CWT[w[0]][t]
    cdt = CDT[d_i][t]
    sumcwt = compute_sumcwt(t,this_sample,CWT)
    sumcdt = CDT[d_i][0] + CDT[d_i][1]
    if w[1] == t:
        cwt -= 1
        cdt -= 1
        sumcwt -= 1
    sumcdt -= 1
    pnum = (cwt+beta) * (cdt+alpha)
    pdenom = (sumcwt + W*beta) * (sumcdt + T*alpha)
    prob = pnum/pdenom
    return prob

def compute_CDT(this_sample):
    """return the CDT. 
       CDT[doc][topic] = number of times 'topic' is assigned to a word in 'doc'.
    >>> this_sample = [[['bank', 1], ['money', 0], ['money', 1], ['bank', 1], ['loan', 0]], 
    ...   [['stream', 1], ['loan', 0], ['river', 1], ['loan', 0], ['money', 1]],
    ...   [['money', 1], ['river', 0], ['bank', 0], ['stream', 1], ['bank', 0]],
    ...   [['bank', 1], ['stream', 1], ['bank', 0], ['bank', 1], ['river', 1]]]
    >>> CDT = compute_CDT(this_sample)
    >>> CDT == [[2,3],[2,3],[3,2],[1,4]]
    True
    """
    CDT = []
    d_i = -1
    for d in this_sample:
        d_i += 1
        CDT.append([])
        for t in range(2):
            cdt = compute_cdt(d_i,t,this_sample)
            CDT[d_i].append(cdt)
    return CDT

def compute_CWT(this_sample):
    """return the CWT.
       CWT[word][topic] = number of times 'word' is assigned to 'topic'.
    >>> this_sample = [[['bank', 1], ['money', 0], ['money', 1], ['bank', 1], ['loan', 0]], 
    ...   [['stream', 1], ['loan', 0], ['river', 1], ['loan', 0], ['money', 1]],
    ...   [['money', 1], ['river', 0], ['bank', 0], ['stream', 1], ['bank', 0]],
    ...   [['bank', 1], ['stream', 1], ['bank', 0], ['bank', 1], ['river', 1]]]
    >>> CWT = compute_CWT(this_sample)
    >>> CWT == {'bank':[3,4],'money':[1,3],'loan':[3,0],'stream':[0,3],'river':[1,2]}
    True
    """
    CWT = {}
    words = set([w[0] for d in this_sample for w in d])
    for w in words:
        CWT[w] = []
        for t in range(2):
            cwt = compute_cwt(w,t,this_sample)
            CWT[w].append(cwt)
    return CWT

def compute_sumcwt(topic,this_sample,CWT):
    """Return the sum of cwt for all words given a topic
    >>> this_sample = [[['bank', 1], ['money', 0], ['money', 1], ['bank', 1], ['loan', 0]], 
    ...   [['stream', 1], ['loan', 0], ['river', 1], ['loan', 0], ['money', 1]],
    ...   [['money', 1], ['river', 0], ['bank', 0], ['stream', 1], ['bank', 0]],
    ...   [['bank', 1], ['stream', 1], ['bank', 0], ['bank', 1], ['river', 1]]]
    >>> CWT = compute_CWT(this_sample)
    >>> compute_sumcwt(0,this_sample,CWT)
    8
    >>> compute_sumcwt(1,this_sample,CWT)
    12
    """
    words = set([w[0] for d in this_sample for w in d])
    a = 0
    for w in words:
        a += CWT[w][topic]
    return a

def compute_cwt(word,topic,this_sample):
    """Return the number of times the word w is assigned to the topic:
    >>> this_sample = [[['bank', 1], ['money', 0], ['money', 1], ['bank', 1], ['loan', 0]], 
    ...   [['stream', 1], ['loan', 0], ['river', 1], ['loan', 0], ['money', 1]],
    ...   [['money', 1], ['river', 0], ['bank', 0], ['stream', 1], ['bank', 0]],
    ...   [['bank', 1], ['stream', 1], ['bank', 0], ['bank', 1], ['river', 1]]]
    >>> compute_cwt('bank',1,this_sample)
    4
    >>> compute_cwt('bank',0,this_sample)
    3
    """
    count = 0
    for d in this_sample:
        for w in d:
            if w[0] == word and w[1] == topic:
                count += 1
    return count

def compute_cdt(doc,topic,this_sample):
    """Return the number of times the topic is assigned to a word in the document:
    >>> this_sample = [[['bank', 1], ['money', 0], ['money', 1], ['bank', 1], ['loan', 0]], 
    ...   [['stream', 1], ['loan', 0], ['river', 1], ['loan', 0], ['money', 1]],
    ...   [['money', 1], ['river', 0], ['bank', 0], ['stream', 1], ['bank', 0]],
    ...   [['bank', 1], ['stream', 1], ['bank', 0], ['bank', 1], ['river', 1]]]
    >>> compute_cdt(1,0,this_sample)
    2
    >>> compute_cdt(1,1,this_sample)
    3
    """
    count = 0
    for w in this_sample[doc]:
        if w[1] == topic:
            count += 1
    return count

def rmse(topic_distr,reference):
    "Return the RMSE of the topic distribution against the reference distribution"
    diff = 0
    count = 0
    for d in range(0,len(topic_distr)):
        for w in range(0,len(topic_distr[d])):
            for t in (0,1):
                diff += (topic_distr[d][w][1][t] - reference[d][w][1][t])**2
                count += 1
    return (float(diff)/count)**.5

def print_topic_distr(topic_distr):
    "Print the topic distribution"
    words = set([w[0] for d in topic_distr for w in d])
    widths = dict()
    print("   ",end="")
    for w in words:
        # Compute the max number of instances of the word in a document
        widths[w] = max([len([w1 for w1 in d if w1[0]==w]) for d in topic_distr])
        print("%-*s " % (widths[w]*4,w),end="")
    print()
        
    for d in topic_distr:
##        print("%3i" % len([t for (w,t) in d if t[0]>=0.5]),end="")
##        print("%3i" % len([t for (w,t) in d if t[1]>=0.5]),end="")
        print("   ",end="")
        for w in words:
            topics = [t for (wi,t) in d if wi==w]
            for t in topics:
                print("%1.1f " % (t[1]),end="")
            for i in range(0,widths[w]-len(topics)):
                print("    ",end="")
            print(" ",end="")
        print()
    print()

def sample(probDict):
    """samples n items from the distribution probDict"""
    nrand = random.random()
    accum = 0
    for (event, prob) in probDict.items():
        accum += prob
        if nrand<accum:
            return event
    raise Exception("Not a valid probability distribution?!?")


if __name__ == '__main__':
    from matplotlib import pyplot
    import doctest
    import sys
    doctest.testmod()

    try:
        arg1 = sys.argv[1]
    except IndexError:
        arg1 = "notest"

    if arg1 == "test":
        print("Handworked example")
        data = [["word0","word0","word0","word0"],
                ["word1","word1","word1","word1"]]
        topic_distr,this_sample = initialise(data)
        print("Initial distribution:")
        print_topic_distr(topic_distr)
        all_rmse = []
        prev_topic_distr = topic_distr
        for i in range(10):
            topic_distr,this_sample = gibbs_iterate(this_sample,1)
            all_rmse.append(rmse(topic_distr,prev_topic_distr))
            # if i % 1 == 0:
            #     print("After iteration %i:" % (i+1))
            #     print_topic_distr(topic_distr)
        print("Final distribution:")
        print_topic_distr(topic_distr)
        pyplot.plot(all_rmse,label="$\sqrt{\sum_{d,w,t}(topic(d,w,t)_i - topic(d,w,t)_{i-1})^2/n}$")
        pyplot.legend()
        pyplot.show()
        sys.exit()
        
    try:
        iterations = int(sys.argv[1])
    except IndexError:
        iterations = 5    

    try:
        modulo = int(sys.argv[2])
    except IndexError:
        modulo = 1    


    words = ("river","stream","bank","money","loan") # All words
    phi = ({'money':1/3.0, 'loan':1/3.0, 'bank':1/3.0},    # word distr per topic
           {'river':1/3.0, 'stream':1/3.0, 'bank':1/3.0})
    docs = ((1,0),(1,0),(1,0),(1,0),(1,0),(1,0),
            (0.5,0.5),(0.5,0.5),(0.5,0.5),(0.5,0.5),(0.5,0.5),
            (0,1),(0,1),(0,1),(0,1))       # topic distr per document
    data = generate_docs(words,phi,docs)
    print("Generated %i documents" % (len(data)))
    topic_distr,this_sample = initialise(data)
    print("Initial distribution:")
    print_topic_distr(topic_distr)
    all_rmse = []
    prev_topic_distr = topic_distr

    for i in range(0,iterations):
        topic_distr,this_sample = gibbs_iterate(this_sample,1)
        all_rmse.append(rmse(topic_distr,prev_topic_distr))
        prev_topic_distr = topic_distr
        if i % modulo == 0:
            print("After iteration %i:" % (i+1))
            print_topic_distr(topic_distr)
    
    print("Final distribution:")
    print_topic_distr(topic_distr)

    pyplot.plot(all_rmse,label="$\sqrt{\sum_{d,w,t}(topic(d,w,t)_i - topic(d,w,t)_{i-1})^2/n}$")
    pyplot.legend()
    pyplot.show()
