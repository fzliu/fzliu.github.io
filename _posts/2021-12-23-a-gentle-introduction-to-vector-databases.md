---
layout: post
title: "A Gentle Introduction to Vector Databases"
description: "An introduction to the vector database: a new type of database purpose-built to for the machine learning era."
categories: ["blog"]
tags: machine-learning
redirect_from: /2021/12/23/a_gentle_introduction_to_vector_databases.html
canonical_url: https://zilliz.com/learn/what-is-vector-database
---

_Update: An earlier version of this post was cross-published to the [Zilliz learning center](https://zilliz.com/learn), [Medium](https://milvusio.medium.com/what-are-vector-databases-8100178c5774), and [DZone](https://dzone.com/articles/what-are-vector-databases)._

_If you have any feedback, feel free to connect with me on [Twitter](https://twitter.com/frankzliu) or [Linkedin](https://www.linkedin.com/in/fzliu/). If you enjoyed this post and want to learn a bit more about vector databases and embeddings in general, check out the [Towhee](https://github.com/towhee-io/towhee) and [Milvus](https://github.com/milvus-io/milvus) open-source projects. Thanks for reading!_

----

In this blog post, I'll introduce concepts related to the vector database, a new type of technology designed to store, manage, and search embedding vectors. Vector databases are being used in an increasingly large number of applications, including but not limited to image search, recommender system, text understanding, video summarization, drug discovery, stock market analysis, and much more.

#### Relational is not enough

Data is everywhere. In the early days of the internet, data was mostly structured, and could easily be stored and managed in relational databases. Take, for example, a book database:

| ISBN       | Year |                 Name                 |    Author     |
| ---------- | ---- | ------------------------------------ | ------------- |
| 0767908171 | 2003 | A Short History of Nearly Everything | Bill Bryson   |
| 039516611X | 1962 | Silent Spring                        | Rachel Carson |
| 0374332657 | 1998 | Holes                                | Louis Sachar  |
| ...

Storing and searching across table-based data such as the one shown above is exactly what relational databases were designed to do. In the example above, each row within the database represents a particular book, while the columns correspond to a particular category of information. When a user looks up book(s) through an online service, they can do so through any of the column names present within the database. For example, querying over all results where the author name is Bill Bryson returns all of Bryson's books.

As the internet grew and evolved, unstructured data (magazine articles, shared photos, short videos, etc.) became increasingly common. Unlike structured data, there is no easy way to store the contents of unstructured data within a relational database. Imagine, for example, trying to search for similar shoes given a collection of shoe pictures from various angles; this would be impossible in a relational database since understanding shoe style, size, color, etc... purely from the image's raw pixel values is impossible.

This brings us to vector databases. The increasing ubiquity of unstructured data has led to a steady rise in the use of machine learning models trained to understand such data. `word2vec`, a natural language processing (NLP) algorithm which uses a neural network to learn word associations, is a well-known early example of this. The `word2vec` model is capable of turning single words (in a variety of languages, not just English) into a list of floating point values, or vectors. Due to the way models is trained, vectors which are close to each other represent words which are similar to each other, hence the term _embedding vectors_. We'll get into a bit more detail (with code!) in the next section.

Armed with this knowledge, it's now clear what vector databases are used for: searching across images, video, text, audio, and other forms of unstructured data via their _content_ rather than keywords or tags (which are often input manually by users or curators). When combined with powerful machine learning models, vector databases have the capability of revolutionizing semantic search and recommendation systems.

| Data UID[^1] | Vector representation                    |
| -------------------- | ---------------------------------------- |
| 00000000             | [-0.31,  0.53, -0.18, ..., -0.16, -0.38] |
| 00000001             | [ 0.58,  0.25,  0.61, ..., -0.03, -0.31] |
| 00000002             | [-0.07, -0.53, -0.02, ..., -0.61,  0.59] |
| ...

In the upcoming sections, I'll share some information about why embedding vectors can be used to represent unstructured data, go over algorithms for indexing and searching across vector spaces, and present some key features a modern vector database must implement.

#### `x2vec`: A new way to understand data

The idea of turning a piece of unstructured data into a list of numerical values is nothing new[^2]. As deep learning gained steam in both academic and industry circles, new ways to represent text, audio, and images came to be. A common component of all these representations is their use of embedding vectors generated by trained deep neural networks. Going back to the example of `word2vec`, we can see that the generated embeddings contain significant semantic information.

__Some prep work__

Before beginning, we'll need to install the `gensim` library and load a `word2vec` model.


```shell
% pip install gensim --disable-pip-version-check
% wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
% gunzip GoogleNews-vectors-negative300.bin
```

    Requirement already satisfied: gensim in /Users/fzliu/.pyenv/lib/python3.8/site-packages (4.1.2)
    Requirement already satisfied: smart-open>=1.8.1 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from gensim) (5.2.1)
    Requirement already satisfied: numpy>=1.17.0 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from gensim) (1.19.5)
    Requirement already satisfied: scipy>=0.18.1 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from gensim) (1.7.3)
    --2022-02-22 00:30:34--  https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.20.165
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.20.165|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1647046227 (1.5G) [application/x-gzip]
    Saving to: ‘GoogleNews-vectors-negative300.bin.gz’

    GoogleNews-vectors- 100%[===================>]   1.53G  2.66MB/s    in 11m 23s

    2022-02-22 00:41:57 (2.30 MB/s) - ‘GoogleNews-vectors-negative300.bin.gz’ saved [1647046227/1647046227]

    gunzip: GoogleNews-vectors-negative300.bin: unknown suffix -- ignored


Now that we've done all the prep work required to generate word-to-vector embeddings, let's load the trained `word2vec` model.


```python
>>> from gensim.models import KeyedVectors
>>> model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
```

__Example 0: Marlon Brando__

Let's take a look at how `word2vec` interprets the famous actor Marlon Brando.


```python
>>> print(model.most_similar(positive=['Marlon_Brando']))
```

    [('Brando', 0.757453978061676), ('Humphrey_Bogart', 0.6143958568572998), ('actor_Marlon_Brando', 0.6016287207603455), ('Al_Pacino', 0.5675410032272339), ('Elia_Kazan', 0.5594002604484558), ('Steve_McQueen', 0.5539456605911255), ('Marilyn_Monroe', 0.5512186884880066), ('Jack_Nicholson', 0.5440199375152588), ('Shelley_Winters', 0.5432392954826355), ('Apocalypse_Now', 0.5306933522224426)]


Marlon Brando worked with Al Pacino in The Godfather and Elia Kazan in A Streetcar Named Desire. He also starred in Apocalypse Now.

__Example 1: If all of the kings had their queens on the throne__

Vectors can be added and subtracted from each other to demo underlying semantic changes.


```python
>>> print(model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))
```

    [('queen', 0.7118193507194519)]


Who says engineers can't enjoy a bit of dance-pop now and then?

__Example 2: Apple, the company, the fruit, ... or both?__

The word "apple" can refer to both the company as well as the delicious red fruit. In this example, we can see that Word2Vec retains both meanings.


```python
>>> print(model.most_similar(positive=['samsung', 'iphone'], negative=['apple'], topn=1))
>>> print(model.most_similar(positive=['fruit'], topn=10)[9:])
```

    [('droid_x', 0.6324754953384399)]
    [('apple', 0.6410146951675415)]


"Droid" refers to Samsung's first 4G LTE smartphone ("Samsung" + "iPhone" - "Apple" = "Droid"), while "apple" is the 10th closest word to "fruit".

#### Generating embeddings with [Towhee](https://towhee.io)

Vector embeddings are not just limited to natural language. In the example below, let's generate embedding vectors for three different images, two of which have similar content:

__Prep work__

For this example, we'll be using [Towhee](https://towhee.io), a framework for developing and running embedding pipelines which include deep learning models (built on top of PyTorch and Tensorflow). We'll also download three images from the [YFCC100M dataset](http://www.yfcc100m.org) to test our embeddings on.


```shell
% pip install towhee --disable-pip-version-check
```

    Requirement already satisfied: towhee in /Users/fzliu/.pyenv/lib/python3.8/site-packages (0.4.0)
    Requirement already satisfied: pyyaml>=5.3.0 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from towhee) (5.4.1)
    Requirement already satisfied: tqdm>=4.59.0 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from towhee) (4.62.3)
    Requirement already satisfied: requests>=2.12.5 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from towhee) (2.26.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from requests>=2.12.5->towhee) (1.26.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from requests>=2.12.5->towhee) (2.0.6)
    Requirement already satisfied: idna<4,>=2.5 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from requests>=2.12.5->towhee) (3.2)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/fzliu/.pyenv/lib/python3.8/site-packages (from requests>=2.12.5->towhee) (2021.5.30)


__Generating embeddings__

Now let's use Towhee to generate embeddings for the test images below. The first and second images should be fairly close to each other in embedding space, while the first and third should be further away:

| ![](https://farm6.staticflickr.com/5012/5493808033_eb1dfcd98f_q.jpg) | ![](https://farm1.staticflickr.com/29/60515385_198df3b357_q.jpg) | ![](https://farm2.staticflickr.com/1171/1088524379_7a150cef81_q.jpg) |
| :----: | :----: | :---: |
| `dog0` | `dog1` | `car` |


```python
>>> from towhee import pipeline
>>> p = pipeline('image-embedding')
>>> dog0_vec = p('https://farm6.staticflickr.com/5012/5493808033_eb1dfcd98f_q.jpg')
>>> dog1_vec = p('https://farm1.staticflickr.com/29/60515385_198df3b357_q.jpg')
>>> car_vec = p('https://farm2.staticflickr.com/1171/1088524379_7a150cef81_q.jpg')
```

__Normalize the resulting vector__

```python
>>> import numpy as np
>>> dog0_vec = dog0_vec / np.linalg.norm(dog0_vec)
>>> dog1_vec = dog1_vec / np.linalg.norm(dog1_vec)
>>> car_vec = car_vec / np.linalg.norm(car_vec)
```

__Now let's compute distances__

With the normalized vectors in place, we can now compute an inverted similarity metric using the Euclidean distance between vectors (lower = more similar). Euclidean distance is the most common distance/similarity metric available to vector database users:

```python
>>> import numpy as np
>>> print('dog0 to dog1 distance:', np.linalg.norm(dog0_vec - dog1_vec))
>>> print('dog0 to car distance:', np.linalg.norm(dog0_vec - car_vec))
```

    dog0 to dog1 distance: 0.80871606
    dog0 to car distance: 1.280709

Towhee has a number of [other embedding generation pipelines](https://towhee.io/pipelines) (image embedding, audio embedding, face embedding, etc) as well.

#### Searching across embedding vectors

Now that we’ve seen the representational power of vector embeddings, let’s take a bit of time to briefly discuss indexing the vectors. Like relational databases, vector databases need to be searchable in order to be truly useful — just storing the vector and its associated metadata is not enough. This is called nearest neighbor search, or NN search for short, and alone can be considered a subfield of machine learning and pattern recognition due to the sheer number of solutions proposed.

Vector search is generally split into two components - the similarity metric and the index. The similarity metric defines how the distance between two vectors is evaluated, while the index is a data structure that facilitates the search process. Similarity metrics are fairly straightforward; the most common similarity metric is the inverse of the L2 norm (also known as Euclidean distance). On the other hand, a diverse set of indices exist, each of which has its own set of advantages and disadvantages. I won’t go into the details of vector indices here (that’s a topic for another article); just know that, without them, a querying across vector databases would be excruciatingly slow.

#### Putting it all together

Now that we understand the representational power of embedding vectors and have a good general overview of how vector search works, it’s now time to put the two concepts together — welcome to the world of vector databases. A vector database is purpose-built to store, index, and query across embedding vectors generated by passing unstructured data through machine learning models.

When scaling to huge numbers of vector embeddings, searching across embedding vectors (even with indices) can be prohibitively expensive. Despite this, the best and most advanced vector databases will allow you to insert and search across millions or even billions of target vectors, in addition to specifying an indexing algorithm and similarity metric of your choosing.

Like the production-ready relational databases, vector databases should meet a few key performance targets before they can be deployed in actual production environments:

- __Scalable__: Embedding vectors are fairly small in terms of absolute size, but to facilitate read and write speeds, they are usually stored in-memory (disk-based NN/ANN search is a topic for another blog post). When scaling to billions of embedding vectors and beyond, storage and compute quickly become unmanageable for a single machine. Sharding can solve this problem, but this requires splitting the indexes across multiple machines as well.
- __Reliable__: Modern relational databases are fault-tolerant. Replication allows cloud-native enterprise databases to avoid having single points of failure, enabling graceful startup and shutdown. Vector databases are no different, and should be able to handle internal faults without data loss and with minimal operational impact.
- __Fast__: Yes, query and write speeds are important, even for vector databases. An increasingly common use case for vector databases is processing and indexing input data in real-time. For platforms such as Snapchat and Instagram, which can have hundreds or thousands of new photos uploaded per second, speed becomes an incredibly important factor.

#### Selecting a vector database

With unstructured data being generated at unprecedented rates, the ability to transform, store, and analyze incoming data streams is becoming a pressing need for application developers looking to use AI/ML. There are a number of open-source vector database projects to choose from - [Milvus](https://milvus.io), [Vespa](https://vespa.ai), and [Weaviate](https://weaviate.io) are three commonly deployed solutions. Some of these projects refer to themselves as _vector search engines_ or _neural search engines_, concepts which are functionally equivalent to vector databases.

For my own personal applications, I'll select [Milvus](https://milvus.io) 99% of the time - it's cloud-native and fast[^3]. [Zilliz](https://zilliz.com) will be releasing a managed version of [Milvus](https://milvus.io) later in 2022, so the option of seamlessly upgrading to a managed vector database will soon be available as well.

Open-source vector database projects have the distinct advantage of being community-driven and thoroughly tested (via a number applications ranging from small personal projects to large commercial deployments). For this reason, I _do not_ recommend using a closed-source vector database such as Pinecone at this time.

#### Some final words

That's all folks - hope this post was informative. There's a [vector database subreddit](https://www.reddit.com/r/vectordatabase) if you're interested in learning more vector databases. In the meantime, if you have any questions, comments, or concerns, feel free to leave a comment below. Stay tuned for more!

---

[^1]: <sub>The "Data UID" field in a vector database is a unique identifier for a single unstructured data element and is similar to the [`_id` field in MongoDB](https://www.mongodb.com/docs/manual/reference/bson-types/#std-label-objectid). Some vector databases also accept a unique filename or path as a UID.</sub>

[^2]: <sub>Early computer vision and image processing relied on local feature descriptors to turn an image into a “bag” of embedding vectors – one vector for each detected keypoint. [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), [SURF](https://people.ee.ethz.ch/~surf/eccv06.pdf), and [ORB](http://www.gwylab.com/download/ORB_2012.pdf) are three well-known feature descriptors you may have heard of. These feature descriptors, while useful for matching images with one another, proved to be a fairly poor way to represent audio (via spectrograms) and images.</sub>

[^3]: <sub>Here's [a great comparison](https://farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-i/) between [Milvus](https://milvus.io) and [Weaviate](https://weaviate.io), two of the more popular open-source vector database options today (tldr: Milvus is better).</sub>
