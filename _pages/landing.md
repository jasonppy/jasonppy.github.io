---
redirect_from: /about/
permalink: /
---

<!-- Hi! I'm Puyuan Peng, a third year Computer Science PhD student at [UT Austin](https://www.utexas.edu/), and Research Scientist Intern at FAIR at [Meta](https://ai.meta.com/) NYC. I mainly work on speech/audio recognition, understanding, and generation, usually under multimodal context (e.g. text, vision). I have first-authored publication in ICASSP/Interspeech/ACL/ECCV. **I'll be graduating in the Spring of 2025 and on the job market, please reach out if interested!** -->

Hi! I'm Puyuan Peng (彭浦源), a 4th year (final year) PhD student at [UT Austin](https://www.utexas.edu/) advised by [David Harwath](https://www.cs.utexas.edu/~harwath/).
I mainly work on speech & audio recognition and generation, including:

1. **Generation**: Text-to-Speech, Speech Editing, Video-to-Audio
2. **Recognition**: Automatic Speech Recognition, Speech Translation, Audio-Visual Speech Recognition, Speech-Image Retrieval, Speech Segmentation, Speech Quantization, Speech Representation Learning

**Research Highlights**:

- [VoiceCraft (ACL2024)](https://arxiv.org/pdf/2403.16973), the zero-shot TTS and Speech Editing model, garnered **7.7k stars** on [GitHub](https://github.com/jasonppy/VoiceCraft) within just five months of its release, **trending globally #1**.
- [Audio-Visual Latent Diffusion Model (ECCV2024)](https://arxiv.org/pdf/2406.09272) generates realistic action sounds for silent egocentric videos and demonstrates zero-shot transfer capabilities in VR games.
- [PromptingWhisper (Interspeech2023)](https://arxiv.org/abs/2305.11095) pioneered the application of prompt-based techniques to large speech models for zero-shot tasks such as audio-visual speech recognition and speech translation without fine-tuning.
- Visually Grounded Speech Research ([Interspeech2023](https://arxiv.org/abs/2305.11435), [2022](https://arxiv.org/pdf/2203.15081.pdf), [ICASSP2022](https://arxiv.org/pdf/2109.08186.pdf), [ASRU2023](https://arxiv.org/abs/2310.07654)) sets state-of-the-art performance in speech-image retrieval, zero-resource speech recognition, and data-efficient representation learning; draws parallels to human language development, analyzed at the Annual Meeting of the Cognitive Science Society ([CogSci](https://escholarship.org/content/qt79t028n8/qt79t028n8_noSplash_eb7a0686a1b74591db4bbd04aa34227f.pdf)).


In addition to my advisor, I have the pleasure to work with and learn from many amazing senior researchers, including (in chronological order): [Karen Livescu (TTIC/UChicago)](https://home.ttic.edu/~klivescu/), [Raymond Mooney (UT)](https://www.cs.utexas.edu/~mooney/), [James Glass (MIT)](https://people.csail.mit.edu/jrg/), [Yoon Kim (MIT)](https://people.csail.mit.edu/yoonkim/), [Abdelrahman Mohamed (Rembrand)](https://scholar.google.com/citations?hl=en&user=tJ_PrzgAAAAJ), [Jonathan Le Roux (MERL)](https://www.jonathanleroux.org/), [Shinji Watanabe (CMU)](https://sites.google.com/view/shinjiwatanabe), [Hung-yi Lee (NTU)](https://speech.ee.ntu.edu.tw/~hylee/index.php), [Kristen Grauman (UT/Meta)](https://www.cs.utexas.edu/users/grauman/), [Wei-Ning Hsu (Meta)](https://scholar.google.com/citations?user=N5HDmqoAAAAJ&hl=en) etc.

I have a Master's degree in Statistics from [The University of Chicago](https://stat.uchicago.edu/alumni/ms-alumni/), and a Bachelor's degree in Mathematics from [Beijing Normal University](https://english.bnu.edu.cn/).  

<!-- In my free time, I like to workout and sing.  -->

contact: pyp@utexas.edu  

## Papers 
(The asterisk '\*' denotes equal contribution)  

<ol reversed>
  <li>
    <strong>VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild</strong><br>
    <span style="font-weight: 550;">Puyuan Peng</span>, Po-Yao Huang, Shang-Wen Li, Abdelrahman Mohamed, David Harwath<br>
    <em>ACL, 2024 (Oral)</em><br>
    <a href="/assets/pdfs/VoiceCraft.pdf">pdf</a> <a href="https://jasonppy.github.io/VoiceCraft_web/">website</a> <a href="https://huggingface.co/spaces/pyp1/VoiceCraft_gradio">interactive demo</a> <a href="https://github.com/jasonppy/VoiceCraft">code</a> <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/jasonppy/VoiceCraft">
  </li>
    <li>
    <strong>Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos</strong><br>
    Changan Chen*, <span style="font-weight: 550;">Puyuan Peng*</span>, Ami Baid, Zihui Xue, Wei-Ning Hsu, David Harwath, Kristen Grauman<br>
    <em>ECCV, 2024 (Oral)</em><br>
    <a href="/assets/pdfs/Action2Sound.pdf">pdf</a> <a href="https://vision.cs.utexas.edu/projects/action2sound/">website</a> <a href="https://github.com/ChanganVR/action2sound">code</a> <a href="https://ego4dsounds.github.io/">data</a>
  </li>
  <li>
    <strong>BAT: Learning to Reason about Spatial Sounds with Large Language Models</strong><br>
    Zhisheng Zheng, <span style="font-weight: 550;">Puyuan Peng</span>, Ziyang Ma, Xie Chen, Eunsol Choi, David Harwath<br>
    <em>ICML, 2024</em><br>
    <a href="https://arxiv.org/pdf/2402.01591.pdf">pdf</a> <a href="https://zhishengzheng.com/BAT/">website</a> <a href="https://github.com/zszheng147/Spatial-AST">Spatial-AST code</a> <a href="https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/seld_spatialsoundqa">BAT code</a> 
  </li>
  <li>
    <strong>Neural Codec Language Models for Disentangled and Textless Voice Conversion</strong><br>
    Alan Baade, <span style="font-weight: 550;">Puyuan Peng</span>, David Harwath<br>
    <em>Interspeech, 2024</em><br>
    <a href="/assets/pdfs/textless_NCLM.pdf">pdf</a> <a href="https://github.com/AlanBaade/TextlessVoiceConversionNDU">code</a>
  </li>
  <li>
    <strong>AV-SUPERB: A Multi-Task Evaluation Benchmark for Audio-Visual Representation Models</strong><br>
    Yuan Tseng, Layne Berry*, Yi-Ting Chen*, I-Hsiang Chiu*, Hsuan-Hao Lin*, Max Liu*, <span style="font-weight: 550;">Puyuan Peng*</span>, Yi-Jen Shih*, Hung-Yu Wang*, Haibin Wu*, Po-Yao Huang, Chun-Mao Lai, Shang-Wen Li, David Harwath, Yu Tsao, Shinji Watanabe, Abdelrahman Mohamed, Chi-Luen Feng, Hung-yi Lee<br>
    <em>ICASSP, 2024</em><br>
    <a href="https://arxiv.org/pdf/2309.10787.pdf">pdf</a> <a href="https://github.com/roger-tseng/av-superb">code</a> <a href="https://av.superbbenchmark.org/">website</a>
  </li>
  <li>
  <strong>Dynamic-superb phase-2: A collaboratively expanding benchmark for measuring the capabilities of spoken language models with 180 tasks</strong><br>
  Chien-yu Huang and 78 authors including Puyuan Peng<br>
  <em>arXiv, 2024</em><br>
  <a href="https://arxiv.org/pdf/2411.05361">pdf</a>
  </li>
  <li>
  <strong>SyllableLM: Learning Coarse Semantic Units for Speech Language Models</strong><br>
  Alan Baade, <span style="font-weight: 550;">Puyuan Peng</span>, David Harwath<br>
  <em>arXiv, 2024</em><br>
  <a href="https://arxiv.org/pdf/2410.04029">pdf</a>
  </li>
  <li>
    <strong>Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization</strong><br>
    <span style="font-weight: 550;">Puyuan Peng</span>, Brian Yan, Shinji Watanabe, David Hawarth<br>
    <em>Interspeech, 2023</em><br>
    <a href="https://arxiv.org/pdf/2305.11095.pdf">pdf</a> <a href="https://github.com/jasonppy/promptingwhisper">code</a>
  </li>

  <li>
    <strong>Syllable Discovery and Cross-Lingual Generalization in a Visually Grounded, Self-Supervised Speech Model</strong><br>
    <span style="font-weight: 550;">Puyuan Peng</span>, Shang-Wen Li, Okko Räsänen, Abdelrahman Mohamed, David Harwath<br>
    <em>Interspeech, 2023</em><br>
    <a href="https://arxiv.org/pdf/2305.11435.pdf">pdf</a> <a href="https://github.com/jasonppy/syllable-discovery">code</a>
  </li>

  <li>
    <strong>Style-transfer based Speech and Audio-visual Scene understanding for Robot Action Sequence Acquisition from Videos</strong><br>
    Chiori Hori, <span style="font-weight: 550;">Puyuan Peng</span>, David Harwath, Xinyu Liu, Kei Ota, Siddarth Jain, Radu Corcodel, Devesh Jha, Diego Romeres, Jonathan Le Roux<br>
    <em>Interspeech, 2023</em><br>
    <a href="https://arxiv.org/pdf/2306.15644.pdf">pdf</a>
  </li>

  <li>
    <strong>Audio-Visual Neural Syntax Acquisition</strong><br>
    Cheng-I Jeff Lai*, Freda Shi*, <span style="font-weight: 550;">Puyuan Peng*</span>, Yoon Kim, Kevin Gimpel, Shiyu Chang, Yung-Sung Chuang, Saurabhchand Bhati, David Cox, David Harwath, Yang Zhang, Karen Livescu, James Glass<br>
    <em>ASRU, 2023</em><br>
    <a href="https://arxiv.org/pdf/2310.07654.pdf">pdf</a> <a href="https://github.com/jefflai108/AV-NSL">code</a>
  </li>

  <li>
    <strong>Zero-shot Video Moment Retrieval With Off-the-Shelf Models</strong><br>
    Anuj Diwan*, <span style="font-weight: 550;">Puyuan Peng*</span>, Raymond J. Mooney<br>
    <em>Workshop on Transfer Learning for Natural Language Processing, 2022</em><br>
    <a href="https://arxiv.org/pdf/2211.02178.pdf">pdf</a>
  </li>

  <li>
    <strong>Word Discovery in Visually Grounded, Self-Supervised Speech Models</strong><br>
    <span style="font-weight: 550;">Puyuan Peng</span>, David Harwath<br>
    <em>Interspeech, 2022</em><br>
    <a href="https://arxiv.org/pdf/2203.15081.pdf">pdf</a> <a href="https://github.com/jasonppy/word-discovery">code</a>
  </li>

  <li>
    <strong>MAE-AST: Masked Autoencoding Audio Spectrogram Transformer</strong><br>
    Alan Baade, <span style="font-weight: 550;">Puyuan Peng</span>, David Harwath<br>
    <em>Interspeech, 2022</em><br>
    <a href="https://arxiv.org/pdf/2203.16691.pdf">pdf</a> <a href="https://github.com/AlanBaade/MAE-AST-Public">code</a>
  </li>

  <li>
    <strong>Self-Supervised Representation Learning for Speech Using Visual Grounding and Masked Language Modeling</strong><br>
    <span style="font-weight: 550;">Puyuan Peng</span>, David Harwath<br>
    <em>The 2nd Workshop on Self-supervised Learning for Audio and Speech Processing at AAAI, 2022</em><br>
    <a href="https://arxiv.org/pdf/2202.03543.pdf">pdf</a> <a href="https://github.com/jasonppy/FaST-VGS-Family">code</a>
  </li>

  <li>
    <strong>Fast-Slow Transformer for Visually Grounding Speech</strong><br>
    <span style="font-weight: 550;">Puyuan Peng</span>, David Harwath<br>
    <em>ICASSP, 2022</em><br>
    <a href="https://arxiv.org/pdf/2109.08186.pdf">pdf</a> <a href="https://github.com/jasonppy/FaST-VGS-Family">code</a>
  </li>

  <li>
    <strong>A Correspondence Variational Autoencoder for Unsupervised Acoustic Word Embeddings</strong><br>
    <span style="font-weight: 550;">Puyuan Peng</span>, Herman Kamper, and Karen Livescu<br>
    <em>The 1st Workshop on Self-Supervised Learning for Speech and Audio Processing at NeurIPS, 2020</em><br>
    <a href="https://arxiv.org/pdf/2012.02221.pdf">pdf</a>
  </li>
</ol>

## Talks
May 2024 at [Meta AI](https://ai.meta.com/meta-ai/), New York, USA  
May 2022 at [Developmental Intelligence Laboratory](https://www.la.utexas.edu/users/dil/), Department of Psychology, UT Austin, USA  
Jan 2022 at [Karen Livescu Group](https://home.ttic.edu/~klivescu/),  Toyota Technological Institute at Chicago, USA.  
Jan 2022 at [Cognitive Machine Learning Group](https://cognitive-ml.fr/), Departement d’Etudes Cognitives, Ecole Normale Supérieure, France.  
