---
layout: post
title: "a16z Blogs Are Just Glorified Marketing"
description: "Taking a look at a16z's blog posts."
categories: ["blog"]
tags: venture-capital
---


#### ... glorified marketing for portfolio companies, that is

I came across one of a16z's blog posts on Hacker News today, titled [Emerging Architectures for LLM Applications](https://a16z.com/2023/06/20/emerging-architectures-for-llm-applications). For folks who didn't catch it, here's the tl;dr:

- The emerging LLM stack is composed of several elements centered around data orchestration tools such as Langchain and Llamaindex. Data pipelines, embedding models, vector databases, and queries form the primary input for these orchestration tools.
- The stack is based on in-context learning, where off-the-shelf LLMs are used and their behavior is controlled through prompting and conditioning on contextual data.
- Strategies for prompting LLMs are becoming increasingly complex and are a core differentiating factor for both closed-source and open-source LLMs. Of these LLMs, strategies for GPT-3.5 and GPT-4 are most common, seeing as OpenAI is the current leader.
- AI agents - programmatic runtimes that can reason and plan - excite both developers and researchers alike, but don't work just yet. Most agent frameworks are currently in PoC phase.

Overall, I thought the article was informative, but I was surprised that the section on vector databases mentions neither [Milvus](https://milvus.io) nor [Zilliz](https://zilliz.com), especially since Milvus was mentioned in an older [a16z blog on data and ML infrastructure](https://a16z.com/2020/10/15/emerging-architectures-for-modern-data-infrastructure/):

<div align="center">
  <img src="/img/emerging-architectures-vector-databases.jpg">
</div>
<p style="text-align:center"><sub>Also of note: another Zilliz project (<a href="https://github.com/zilliztech/GPTCache">GPTCache</a>) is listed in the post.</sub></p>

My initial instinct was that Milvus was left off because it is part of the LF AI & Data Foundation rather being a project wholly owned by Zilliz, so I left a comment on the HN post that links back to the Milvus website. I came back a couple of hours later to find an interesting take:

<div align="center">
  <img src="/img/emerging-architectures-comment.jpg">
</div>
<p style="text-align:center"><sub>Full disclosure: we (Zilliz) raised $103M back in 2022, and Pinecone raised $100M this April.</sub></p>

Running it back in my head, I felt that SheepHerdr's response actually made excellent sense - a16z's ultimate goal is to generate returns for LPs, and the best way to do that is by supporting founders and propping their portfolio companies. To me, this is also unequivocally unfair to Vespa, Weaviate, etc as it delivers a subliminal message that they have no realistic long-term chance in the vector database space relative to Pinecone. This, of course, is absolute nonsense: vector databases are _NOT_ a zero-sum game.

I dove a bit deeper and was surprised to find that this is fairly commonplace behavior for a16z as a firm:

- The aforementioned article also lists Databricks in the "Data Pipelines" section, but not [Snowflake](https://www.snowflake.com/blog/building-data-centric-platform-ai-llm/). There is a [Snowflake loader for Langchain](https://github.com/hwchase17/langchain/blob/master/langchain/document_loaders/snowflake_loader.py) and a guide for using [Llamaindex with Snowflake](https://github.com/jerryjliu/llama_index/blob/main/docs/guides/tutorials/sql_guide.md). Databricks is an a16z portfolio company.
- [The Modern Transactional Stack](https://a16z.com/2023/04/14/the-modern-transactional-stack/) doesn't come close to listing all of the available data connectors. To be fair, Airbyte and Fivetran (an a16z portfolio company) are the two largest and most well-known, but to distill the entire segment to just two companies seems unfair.
- a16z's crypto division has backed LayerZero, going as far as actively [voting against Wormhole](https://www.dlnews.com/articles/defi/uniswap-taps-wormhole-for-bridge-to-bnb-in-defeat-for-a16z/), a LayerZero competitor. Side note: LayerZero was also featured in a16z's [Crypto Startup School](https://a16zcrypto.com/posts/announcement/crypto-startup-school-2023-recap-resources/).

These are just three random examples I dug out - there are probably many other examples in verticals that I am unfamiliar with.

#### Other LLM/GenAI Infrastructure landscapes

Here's a couple alternative landscapes that are, in my eyes, more wholly representative:
- [ML/AI/Data Landscape](https://mattturck.com/mad2023) ([Interactive version](https://mad.firstmark.com/)). Matt Turck's MAD Landscape is arguably the most complete out there. Companies that do vector search are listed under "Infrastructure/Vector Database" and "Analytics/Enterprise Search" categories. It was released in February 2023 so it's about 4 months old, but a good resource nonetheless.
- [Future of AI-Native Infrastructure](https://www.unusual.vc/post/ai-native-infrastructure-will-be-open). This one's from Wei Lien Dang and David Hershey of Unusual Ventures. I found this pretty unique as it has a vertical for AI agents. It's unfortunately not as complete as the MAD Landscape (missing Vespa, Vectara, etc), but still a good overview.
- [The New Language Model Stack](https://www.sequoiacap.com/article/llm-stack-perspective/). Sequoia Capital's blog post on the LLM stack is also excellent. Milvus isn't in the diagram, but it's mentioned in the section on vector databases.
- [Vector Database Landscape](https://twitter.com/YingjunWu/status/1667232357953466369/). Yingjun Wu's infographic is centered specifically around vector search infrastructure.

#### Final thoughts

I have tremendous respect for a16z, a firm that helped pioneer the practice of working with and nurturing founders rather than forcing them out pre-IPO or minmaxing term sheets. Their content is also incredibly informative and valuable for understanding the nuances of building a company, from finding PMF to hiring executives. I also wholeheartedly understand a16z's motivation for sharing knowledge and highlighting their portfolio companies, but to do so under the guise of being helpful and impartial is just plain silly. In particular, a16z's blog post yesterday has as much to do with emerging strategies for portfolio company marketing as it does with emerging architectures for LLM applications. This practice would be somewhat analagous to Google putting paid URLs at the very top of search results without an "Ad" label. (To be clear, Google doesn't do this.)

I'd like to end with some glorified marketing of my own:

```shell
% pip install milvus
```
