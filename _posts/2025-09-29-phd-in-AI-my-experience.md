---
title: "PhD in AI – My Experience"
date: 2025-09-29
categories:
  - Story
---
In October 2024, I contacted 9 companies for research positions in AI. I received interview invites from 8 of them, finished full interview loops with 6, and got offers from 4. The total market cap of the 4 companies that offered me a research scientist job was $10 trillion by mid-September 2025. I chose Meta.

I couldn’t have imagined this 4 years ago, when I set foot in the CS department at UT Austin, with no prior CS degree and very little research experience.



## 1. Getting into AI without a CS degree

*Jan 2020 – Dec 2020*

I did my undergrad in math at Beijing Normal University and a master’s in statistics at the University of Chicago.

In my first year of the master’s program, I tried to get a job in data science. I practiced LeetCode but found it too hard and impractical. I tried Spark, scikit-learn, and other data analysis tools and found them too boring. After submitting about 30 applications, only 3 companies sent me online assessments, and none gave me an offer. I realized the field was overly crowded.

At the same time, I took a course offered by Prof. Karen Livescu at TTIC called Speech Technologies and did a successful course project, which convinced Prof. Livescu to continue working with me during the summer of 2020. She eventually recommended me to her academic sibling Prof. David Harwath at UT Austin for PhD studies. Both Karen and David did their PhDs at MIT under Prof. Jim Glass (the lab that produced many of the top speech researchers in the US).

I was shocked when I received the offer letter from the UT Computer Science department because their website explicitly said that their PhD program is very competitive and not recommended for people without a CS background—and I had no CS background. In addition, they required a TOEFL transcript, but I only had an expired IELTS score. With no hope of getting a reply, I emailed the admin. To my surprise, they replied immediately, saying that a barely expired IELTS would work—just email them the transcript and they’d add it for me.



## 2. Preparing for a fast-track PhD

*Apr 2021 – Aug 2021*

After accepting the PhD offer, during my meeting with David I asked him how many years he expected a PhD to take. David said 6 years was the standard, and he himself had taken 7 because he changed research direction during his PhD.

I wasn’t interested in spending 6 years, even though I might need to, since I had to take undergrad CS courses to make up for my non-CS background.

To make up for my weak research background, two months after I accepted the PhD offer, in June 2021, I emailed David for research directions. To my surprise, he replied with a long list of 10 research ideas (which I knew very little about). I tried my best to search for anything I could find before our next meeting, and during the meeting, we decided to go with using Transformers to model visually grounded speech. I picked this topic because:

  1. Transformers had already proven to be the next-generation architecture—low risk and very rewarding for a research newbie like me.
  2. The research area of visually grounded speech was co-created by my advisor David, so I could receive a lot of great feedback from him.

Before I even started working on this project, I spent a month reading the PyTorch Lightning codebase and wrote a simplified version of it for myself. That way, I could avoid writing boilerplate code while having a framework I was very familiar with.

Doing the infrastructure work first turned out to be extremely beneficial—the framework I wrote ended up being used in almost every one of my PhD projects, and as I progressed, I continuously refined it to add new features and make it more flexible.



## 3. First year – sweet academia

*Sep 2021 – May 2022*

Once the infrastructure was done, the project was very straightforward. Just like Transformers were outperforming ConvNets in many other fields, with enough data and proper pretraining, Transformers led to significantly better performance in visually grounded speech as well. I felt like I did very little work, and the project just worked. After about a month, our model started to outperform baselines, and another month later, we further boosted the performance and started writing the paper.

We posted the paper in September 2021—the first month of my PhD!

The wonderful start encouraged me to work harder. I decided not to take weekends off for the first year of my PhD. To force myself to go to campus every day rather than stay in my apartment, I bought an unlimited meal plan at UT. For $1,600 per semester, I got unlimited breakfast, lunch, and dinner at any dining hall. This also helped me avoid cooking or eating out. That was especially important because when I settled into my apartment near UT, I only had $2,000 left in my bank account. And in the first year, my salary after tax was only $2,000 per month, while my rent was $1,100.

The only thing I thought about in my first year was visually grounded speech. One issue with the model I published in September 2021 was that it had many complicated modules, making it difficult for others to use. I therefore proposed simplifying the architecture into a two-tower model: a vision tower and an audio tower. Unfortunately, after a few weeks of tuning, the simplified model could not surpass the old, more complicated one.

The vision transformer I used was called DINO. One intriguing behavior of DINO is that even though it was never trained on semantic labels, its internal representation could segment an image into different semantic parts. Out of curiosity, I looked at the self-attention and internal representation in my audio transformer, and was shocked to see that the model was segmenting out individual syllables and words in continuous speech! The model had never seen words during training. This was very similar to how humans acquire language—by looking and listening.

We summarized the finding and wrote it up into two papers on self-supervised word and syllable discovery. My work on visually grounded speech received a lot of attention, and I got to present at ENS Paris, TTIC, and UT’s Developing Intelligence Lab—all in my first year.


## 4. Second year – the realization and the pivot

*May 2022 – May 2023*

Publishing papers and giving (virtual) talks around the world made my first year feel smooth. What I didn’t expect was to then enter a 6-month stretch where nothing worked.

Hooked on the idea of emergent linguistic unit discovery, I explored every possible method to make models discover linguistic units more accurately without textual supervision. Around this time, David connected me with Abdelrahman Mohamed and Daniel Li from Meta (who later became long-term collaborators and had a huge impact on my life). Even though my connections broadened, my research stalled. During weekly meetings with David, Abdo, and Daniel, I sometimes felt embarrassed that everything I tried had failed.

To make things worse, I realized that citations are often treated as a key measure of PhD achievement. My visually grounded speech papers weren’t being cited much. It was disappointing to see that even though well-known researchers were interested, the work wasn’t spreading widely.

From April to September 2022, I gradually sank into depression. I had published 3 papers in my first year, but they seemed to have little impact. Now I was stuck, and nothing was working. I remember sitting in a chair listening to podcasts or audiobooks for hours, but I couldn’t shake the sense of failure.

Then, at the end of September 2022, OpenAI released their first speech model—Whisper. I immediately realized it was my chance to pivot. Whisper is a speech recognition model trained on web-scale data. Textual Large language models trained on web-scale data had already shown emergent zero-shot capabilities: during inference, you can prompt them to do tasks they weren’t explicitly trained on. Whisper is essentially an audio-conditioned LLM—could we prompt it to perform unseen tasks during inference?

After some trial and error, I found that with carefully designed prompts, Whisper could indeed handle unseen recognition tasks. Since I had less experience in recognition, I contacted Prof. Shinji Watanabe at CMU, who connected me with his student Brian Yan. We started collaborating and quickly landed the paper.

Our Prompting Whisper paper received far more attention from industry than my visually grounded speech work. For the first time, I saw people talking about my research online—without me initiating it.

Wanting to continue the momentum, I tried fine-tuning Whisper to push performance even further. That turned out to be a mistake: fine-tuning wasn’t intellectually interesting, and the massive compute required made iteration painfully slow. Sometimes I’d launch an experiment, wait days for results, and get nothing useful.



## 5. Third year – pivot again and an 8k-GitHub-star model!

*May 2023 – May 2024*

After struggling with finetuning Whisper for a month, I started looking for new directions. Speech recognition was maturing—approaching commercial quality—but speech generation, such as text-to-speech, was lagging behind. Commercial systems needed huge amounts of engineering to work properly.

In January 2023, Microsoft released VALL-E, a large language model for text-to-speech with zero-shot voice cloning. Its elegant, scalable design convinced me: this was the GPT moment for speech.

I decided to dive into speech generation, even though I was the only person in my lab working on it—and my advisor didn’t have much experience in the field either.

To catch up, I spent a month reading papers and reaching out to seasoned researchers for virtual chats. I chose to work on a unified LLM-based model for speech editing and voice cloning text-to-speech. I loved audiobooks and podcasts, and editing—removing filler words or replacing small sections—was a natural use case. Voice cloning TTS could be seen as a subtask of editing, so it made sense to unify them.

The project wasn’t easy. Being the only person in the lab working in a new area meant solving a lot of hidden problems that papers didn’t explain. I spent 8 months on it. For example, the baseline model wasn’t trained on large-scale data. To compare fairly, I had to retrain it on our data—but its code wasn’t scalable. I had to rewrite it entirely. It was also my first time coding up a language model from scratch, and small mistakes, like misplacing a special token, could break the whole system.

By January 2024, the model—VoiceCraft—was ready. We quickly realized there was nothing else as powerful. Instead of just publishing, we did three things that turned VoiceCraft into a brand:

  1. Open-sourced the code and weights
  2. Provided ongoing support for developers worldwide through GitHub
  3. Built polished demos and shared them widely

That made all the difference. VoiceCraft went viral: 8,000+ GitHub stars, posts with hundreds of thousands of views, and even a subreddit created by the community. Someone posted on Reddit, “VoiceCraft: I’ve never been more impressed in my entire life!” Famous figures like Marc Andreessen followed me, and VCs from Sequoia and Microsoft Ventures reached out.

For two months, a community grew around VoiceCraft—building demos, making it accessible to non-technical users.

VoiceCraft became the first open-sourced LLM-based text-to-speech research project. Its success inspired more researchers to open-source their own models, sparking a boom in speech generation research.

Afterward, I did an internship at Meta NYC with Wei-Ning Hsu. (I’ll share more about that in another post.)


## 6. Fourth year – the busiest year of my life

*Sep 2024 – Apr 2025*

When I returned, I was in my fourth and final year.

The fall of 2024 was my busiest semester: I was juggling 4 things at once—1) thesis writing, 2) job search and interviews, 3) my final project at school, and 4) my internship project at Meta.

From mid-September to late November, my days started at 7:30. I’d read a mantra I wrote to prepare myself mentally, then go to the gym to prepare myself physically. Then the real work began.

Interviews were the hardest part. Early on, I bombed both coding and behavioral interviews. For coding, I wasn’t prepared—by choice. I decided not to grind LeetCode, because the problems felt disconnected from real research. I also didn’t have time, with my thesis and projects demanding attention. Looking back, now that I’ve gone through 50 interviews, I’d say: if you can make time, practice LeetCode as much as possible.

For non-technical interviews, I didn’t realize how much doing PhD had eroded my communication skills. I often said awkward things. Luckily, I kept notes reflecting on every interview, and gradually improved.

I usually stopped working after 7, went for a 40-minute walk, had dinner, listened to audiobooks, read my mantra again, and slept before midnight.

In the end, 4 out of 8 companies gave me offers. As it turned out, even if you bomb coding interviews, they’ll still hire you if your research skills stand out.

By early December, I started negotiating offers. For the first time that semester, I felt I didn’t need my mantra every morning and night for strength.

My final semester in spring 2025 was still busy, but much more relaxed. I went out with friends more and savored the achievements of my PhD. Time flew. On April 1st, I officially graduated.



## Looking back

From arriving at UT with no CS background and just $2,000 in my bank account, to creating 8 thousand start GitHub repo and joining Meta—I could never have predicted this journey.

A PhD isn’t just about publishing papers. It’s about persistence through failure, learning when to pivot, and building something that excites others.

If you’re considering a PhD or a career in AI: you don’t need to have it all figured out at the start. Just begin, keep learning, and adapt along the way. The opportunities will come.

