---
layout: post
title: "Deep Dive Into Embeddings"
description: "A deep dive into embeddings, embedding vectors, and similarity search."
categories: ["blog"]
tags: machine-learning
---

I've broached the subject of embeddings/embedding vectors in prior blog posts on [vector databases](/blog/a-gentle-introduction-to-vector-databases) and [ML application development](/blog/making-machine-learning-more-accessible-for-application-developers), but haven't yet done a deep dive on embeddings and some of the theory behind how embedding models work. As such, this article will be dedicated towards going a bit more in-depth into embeddings/embedding vectors, along with how they are used in modern ML algorithms and pipelines.

A quick note - this article will require an intermediate knowledge of deep learning and neural networks. If you're not quite there yet, I recommend first taking a look at [Google's ML Crash Course](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture). The course contents  are great for understanding the basics of neural networks for computer vision and machine learning.

#### A quick recap

Vectorizing data via embeddings[^1] is, at its heart, a method for _dimensionality reduction_. Traditional dimensionality reduction methods - [PCA](https://towardsdatascience.com/the-most-gentle-introduction-to-principal-component-analysis-9ffae371e93b), [LDA](https://www.mygreatlearning.com/blog/understanding-latent-dirichlet-allocation/), etc. - use a combination of linear algebra, kernel tricks, and other statistical methods to "compress" data. On the other hand, modern deep learning models perform dimensionality reduction by mapping the input data into a _latent space_, i.e. a representation of the input data where nearby points are correspond to semantically similar data points. What used to be a one-hot vector representing a single word or phrase, for example, can now be represented as a dense vector with a significantly lower dimension. We can see this in action with the [Towhee library](https://github.com/towhee-io/towhee):

```shell
% pip install towhee  # pip3
% python              # python3
```

```python
>>> import towhee
>>> text_embedding = towhee.dc(['Hello, world!']) \
...     .text_embedding.transformers(model_name='distilbert-base-cased') \
...     .to_list()[0]
...
>>> embedding  # includes punctuation and start & end tokens
```

    array([[ 0.30296388,  0.19200979,  0.10141158, ..., -0.07752968,  0.28487974, -0.06456392],
           [ 0.03644813,  0.03014304,  0.33564508, ...,  0.11048479,  0.51030815, -0.05664057],
           [ 0.29160976,  0.43050566,  0.46974635, ...,  0.22705288, -0.0923526 , -0.04366254],
           [ 0.14108554, -0.00599108,  0.34098792, ...,  0.16725197,  0.10088076, -0.06183652],
           [ 0.35695776,  0.30499873,  0.400652  , ...,  0.20334958,  0.37474275, -0.19292705],
           [ 0.6206475 ,  0.50192136,  0.602711  , ..., -0.03119299,  1.1860386 , -0.6167787 ]], dtype=float32)


Embedding algorithms based on deep neural networks are almost universally considered to be stronger than traditional dimensionality reduction methods. These embeddings are being used more and more frequently in the industry in a variety of applications, e.g. content recommendation, question-answering, chatbots, etc. As we'll see later, using embeddings to represent images and text _within_ neural networks has also become increasingly popular in recent years.

<div id="embedding-viz"></div>
<p style="text-align:center"><sub>Visualizing text embeddings produced by <a href="https://towhee.io/text-embedding/transformers">DistilBERT</a>. Note how "football" is significantly closer to "soccer" than it is to "footwear" despite "foot" being common both words.</sub></p>

#### Supervised embeddings

So far, my previous articles have used embeddings from models trained using _supervised learning_, i.e. neural network models which are trained from labelled/annotated datasets. The [ImageNet](https://image-net.org/) dataset, for example, contains a curated set of image-to-class mappings, while question-answering datasets such as [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) provide 1:1 sentence mappings in different languages.

Many well-known models trained across labelled data use cross-entropy loss or mean-squared error. Since the end goal of supervised training is to more or less replicate 1:1 mappings between input data and annotations (e.g. output a class token probability given an input image), embeddings generated from supervised models seldom use the output layer. The standard ResNet50 model trained across ImageNet-1k, for example, outputs a 1000-dimension vector corresponding to probabilities that the input image is an instance of the _N_<sup>th</sup> class label.

<div align="center">
  <img align="center" src="https://d2l.ai/_images/lenet-vert.svg">
</div>
<p style="text-align:center"><sub>LeNet-5, one of the earliest known neural network architectures for computer vision. Image by <a href="https://d2l.ai/">D2L.ai</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</sub></p>

Instead, most modern applications use the penultimate layer of activations as the embedding. In the image above (LeNet-5), this would correspond to the activations between the layers labelled `FC (10)` (10-dimensional output layer) and `FC (84)`. This layer is close enough to the output to accurate represent the semantics of the input data while also being a reasonably low dimension. I've also seen computer vision applications which use pooled activations from a much earlier layer in the model. These activations capture lower-level features of the input image (corners, edges, blogs, etc...), which can result in improved performance for tasks such as logo recognition.

#### Encoders and self-supervision

A major downside of using annotated datasets is that they require annotations (this sounds a bit idiotic, but bear with me here). Creating high-quality annotations for a particular set of input data requires hundreds if not thousands of hours of curation by one or many humans. The full ImageNet dataset, for example, contains approximately 22k categories and required an army of 25000 humans to curate. To complicate matters further, many labelled datasets contain often times unintended inaccuracies, flat-out errors, or [NSFW content](https://github.com/vinayprabhu/Dataset_audits/blob/master/Notebooks/ImageNet_2_NSFW_analysis.ipynb) in the curated results. As the number of such instances increases, the quality of embeddings generated by an embedding model trained with supervised learning decreases significantly.

<div align="center">
  <img align="center" src="https://source.unsplash.com/jSjgQmaQtVQ">
</div>
<p style="text-align:center"><sub>An image tagged as "Hope" from the <a href="https://unsplash.com/data">Unsplash dataset</a>. To humans, this is a very sensible description, but it may cause a model to learn the wrong types of features during training. Photo by <a href="https://unsplash.com/photos/jSjgQmaQtVQ">Lukas L</a>.</sub></p>

Models trained in an _unsupervised_ fashion, on the other hand, do not require labels. Given the insane amount of text, images, audio, and video generated on a daily basis, models trained using this method essentially have access to an infinite amount of training data. The trick here is developing the right type of models and training methodologies for leveraging this data. An incredibly powerful and increasing popular way of doing this is via _autoencoders_ (or _encoder/decoder_ architectures in general).

_Autoencoders_ generally have two main components. The first component is an encoder: it takes some piece of data as the input and transforms it into a fixed-length vector. The second component is a decoder: it maps the vector back into the original piece of data. This is known as an encoder-decoder architecture[^1]:

<div align="center">
  <img align="center" src="https://d2l.ai/_images/encoder-decoder.svg">
</div>
<p style="text-align:center"><sub>Illustration of encoder-decoder architecturs. Image by <a href="https://d2l.ai/">D2L.ai</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</sub></p>

The blue boxes corresponding to the `Encoder` and `Decoder` are both feed-forward neural networks, while the `State` is the desired embedding. There's nothing paricularly special about either of these networks - many image autoencoders will use a standard ResNet50 or ViT for the encoder network and a similarly large network for the decoder.

Since autoencoders are trained to map the input to a latent state and then back to the original data, unsupervised or self-supervised embeddings are taken directly from the output layer of an encoder as opposed to an intermediate layer for models trained with full supervision. As such, autoencoder embeddings are only meant to be used to for _reconstruction_. In other words, they can be used to represent the input data but are generally not powerful enough to represent semantics, such as what differentiates a photo of a cat from an photo of a dog.

In recent years, numerous improvements to self-supervision beyond the traditional autoencoder have come about[^3]. For NLP, pre-training models with _context_, i.e. where a word or character appears relative to others in the same sentence or phrase, is commonplace and is now considered the de-facto technique for training state-of-the-art text embedding models[^4]. Self-supervised computer vision embedding models are also roaring to life; contrastive training techniques[^5] that rely on data augmentation have shown great representational power

#### Embeddings as an input to other models

Embedding models are highly unique; not only are they valuable for generic application development, but their outputs are often used in other machine learning models. A great example of this is OpenAI's [CLIP](https://github.com/openai/CLIP), a large neural network model that is trained to match images with natural langugage. CLIP is trained on what amounts to essentially infinite data from the internet, e.g. Flickr photos and the corresponding photo title.

<div align="center">
  <img align="center" src="https://github.com/openai/CLIP/raw/main/CLIP.png">
</div>
<p style="text-align:center"><sub>Encoders and their corresponding embeddings are used to great effect in OpenAI's CLIP model. Image by <a href="https://github.com/openai">OpenAI</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</sub></p>

CLIP is used as a core component in DALLE (and DALLE-2), a text-to-image generation engine from OpenAI. There's been a lot of buzz around DALLE-2 recently, but none of that would be possible without CLIP's representational power.

While DALLE-2's results have been impressive, image embeddings still have significant room for growth. Certain higher-level semantics, such as numbers or  are still difficult to represent in an eimage embedding. , is still difficult

#### Generating your own embeddings

I've pointed out the [Towhee](https://github.com/towhee-io/towhee) open-source project a couple of times in the past, showing how it can be used to generate embeddings as well as develop applications that require embeddings. Towhee wraps over a hundred image embedding and classification models from a variety of sources ([`timm`](https://github.com/rwightman/pytorch-image-models), [`torchvision`](https://github.com/pytorch/vision), etc...) and trained with a variety of different techniques. Towhee also has many NLP models as well, courtesy of 🤗's [Transformers](https://github.com/huggingface/transformers) library.

Let's take a step back and peek at how Towhee generates these embeddings under the hood.

```python
>>> from towhee import pipeline
>>> p = pipeline('image-embedding')
>>> embedding = p('towhee.jpg')
```

With [Towhee](https://github.com/towhee-io/towhee), the default method for generating embeddings from a supervised model is simply to remove the final classification or regression layer. For PyTorch models, we can do so with the following example code snippet:

```python
>>> import torch.nn as nn
>>> import torchvision
>>> resnet50 = torchvision.models.resnet50(pretrained=True)
>>> resnet50_emb = nn.Sequential(*(list(resnet50.children())[:-1]))
```

The last line in the above code snippet recreates a feed-forward network (`nn.Sequential`) composed of all layers in `resnet50` (`resnet50.children()`) with the exception of the final layer (`[:-1]`). Intermediate embeddings can also be generated with the same layer removal method. This step is unnecessary for models trained with contrastive/triplet loss or as an autoencoder.

Models based on `timm` and `transformers` also maintain their own methods which make feature extraction easy:

```python
>>> import timm
>>> model = timm.models.resnet50(pretrained=True)
>>> emb = model.forward_features(img)
```

```python
>>> import transformers
>>> model = AutoModel.from_pretrained('bert-base-uncased')
>>> out = model('Hello, world!')
>>> emb = out.last_hidden_state()
```

Towhee maintains wrappers for both `timm` and `transformers`:

```python
>>> import towhee
>>> text_embedding = towhee.dc(['Hello, world!']) \
...     .text_embedding.transformers(model_name='distilbert-base-cased') \
...     .to_list()[0]  # same as intro example
...
>>> img_embedding = towhee.glob('./towhee.jpg') \
...    .image_decode() \
...    .image_embedding.timm(model_name='resnet50') \
...    .show()
...
```

#### Other resources

I've compiled a list of other great resources on embeddings, if you want to read more or look at some other perspectives:

1. The [Milvus documentation](https://milvus.io/docs/v1.1.0/vector.md) provides an overview of embedding vectors (as it relates to storage).

2. OpenAI maintains page on [text embeddings](https://beta.openai.com/docs/guides/embeddings) that you can check out.

3. [Will Koehrsen](https://willkoehrsen.github.io/) provides a [great overview of embeddings](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526). It's a bit dated (2018), but still a great resource.

4. See how embeddings are being used at [Twitter](https://blog.twitter.com/engineering/en_us/topics/insights/2018/embeddingsattwitter) for recommendation.

5. [My previous post](/blog/a-gentle-introduction-to-vector-databases) on vector databases introduces embedding storage components.

---

[^1]: Adapted from [D2L.ai](https://github.com/d2l-ai/d2l-en). [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

[^2]: TODO

[^3]: I won't cover all of these improved training techniques. The past couple of years have seen enough ML novelties to fill an entire book!

[^4]: Patches or crops of whole images do techically have context as well. Algorithms and models which understand context are crucial to the subfield of computer vision/graphics known as _inpainting_. Pre-training techniques for computer vision appliations hasn't been as successful, but it will likely become more viable in the near future. [This 2021 paper](https://arxiv.org/abs/2111.06377) shows how masking patches in an autoencoder can be used for pre-training vision transformers.

[^5]: [SimCLR](https://arxiv.org/abs/2002.05709) and [data2vec](https://arxiv.org/abs/2202.03555) both make use of masking and/or other augmentations for self-supervised training.


<script src="https://d3js.org/d3.v6.js"></script>
<script>

class Embedding {
  constructor(name, xval, yval) {
    this.name = name;
    this.xval = xval;
    this.yval = yval;
  }
}

const margin = {top: 10, right: 10, bottom: 40, left: 40},
      width = 360 - margin.left - margin.right,
      height = 360 - margin.top - margin.bottom;

const svg = d3.select("#embedding-viz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

// embedding data
data = [new Embedding("football", 1.53, 0.08),
        new Embedding("footwear", -1.19, 0.65),
        new Embedding("soccer", 1.27, 0.13)];

const x = d3.scaleLinear()
  .domain([-2, 2])
  .range([0, width]);
svg.append("g")
  .attr("transform", "translate(0," + height + ")")
  .call(d3.axisBottom(x));

const y = d3.scaleLinear()
  .domain([-2, 2])
  .range([height, 0]);
svg.append("g")
  .call(d3.axisLeft(y));

// tooltip for text
const tooltip = d3.select("#embedding-viz")
  .append("div")
  .style("opacity", 0)
  .attr("class", "tooltip")
  .style("background-color", "white")
  .style("border", "solid")
  .style("border-width", "1px")
  .style("border-radius", "5px")
  .style("padding", "10px");

// dots
svg.append("g")
  .selectAll("dot")
  .data(data)
  .enter()
  .append("circle")
    .attr("cx", function(d) { return x(d.xval); } )
    .attr("cy", function(d) { return y(d.yval); } )
    .attr("r", 7)
    .style("fill", "#69b3a2")
    .style("opacity", 0.3)
    .style("stroke", "white");

// text
svg.append("g")
  .selectAll("text")
  .data(data)
  .enter()
  .append("text")
    .attr("x", function(d) {return x(d.xval)+5})
    .attr("y", function(d) {return y(d.yval)-5})
    .text(function(d) { return d.name })
    .attr("font-size", "10px");

</script>
