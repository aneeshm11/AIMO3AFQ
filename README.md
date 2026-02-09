[Math Corpus Prize] Activation Aware Fusion

# 1. Motivation

Current open weight models achieve strong performance on standard math benchmarks, indicating that existing datasets are increasingly saturated. This dataset has two main subsets. It introduces new, harder math problems to increase difficulty and novelty via a fusion algorithm to generate new questions. And another subset which focuses on generating high-quality solutions for existing problems using large budget reasoning with maximum capacity to align the solutions for problems in existing datasets with the newly released models. 

---

# 2.  Activation-Aware Fusion Algorithm

The goal of this algorithm is to generate new math problems that remain internally coherent for a model, rather than synthetically combining questions in ways that look reasonable to humans but confuse the model. Naive fusion fails. A straightforward approach would be to randomly pair two questions or to group them by topic labels such as “algebra” or “number theory.” This often produces invalid or unstable problems. Two questions that appear similar to a human can be very far apart in the model’s internal representation space, leading to hallucination, over-reasoning to land at the solution of the fused problem in this case.

Even clustering-based methods such as k-means are not ideal in this setting. The representation space is high dimensional (GPT OSS models have hidden size of 2880), the number of questions are large and these methods produced very low convergence and risked producing sub optimal results.

For this reason, fusion is preferred to be dependent on the model’s own representations in the following steps

## 2.1  Representation and similarity

Each question is passed through [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) without generating a solution. The final hidden state representation (or embeddings in simple terms) of the question (without the prompt) is extracted and stored. All representations are L2-normalized, and cosine similarity is computed between every pair of questions. For each question, all other questions are ranked from most similar to least similar, producing an ordered neighbor list per question. This similarity is not always semantic in the human sense, but reflects how the model internally interprets and encodes the question.

## 2.2 Iterative fusion strategy

A greedy strategy such as, pairing each question with its nearest neighbor and not using it further and removing them both would immediately collapse the dataset to half its size and heavily bias questions. For example if we pick 10 questions and keep fusing, the last 2 questions are forced to fuse even though they might not have sufficient similarity. 

Instead, an iterative traversal is used.

The algorithm runs for multiple iterations. In iteration 1, each question attempts fusion with its most similar neighbor. In iteration 2, it attempts fusion with its second most similar neighbor, and so on. But after this the question is not discarded, but used again. To prevent over-representation, each question is allowed to appear in at most K fusion pairs. Once a question reaches this cap, it is skipped in future iterations. This ensures that early questions do not dominate the dataset and that later questions are not starved of potential fusion partners.

Before accepting a fusion pair (i,j) the reverse pair (j,i) is checked to avoid duplicates.

## 2.3 Fusion generation

Since the fusion dataset comes from a main dataset (details of which will be discussed further), each fusion pair is classified into one of two types:

```Fusion pair types```

```AA```: both questions are solvable by the model

```AB```: only one question is solvable by the model

Pairs where both questions are unsolved (BB) are ignored. This acts as a simple and effective difficulty control mechanism which I chose for now. 

For each accepted fusion pair (i, j) the following are provided to the generator model:

- The two original questions (qi, qj)
- Their corresponding final answers (ai, aj)

GPT-OSS-120B is then used as a generator model in high-reasoning mode with a large token budget (≈150k tokens per generation). The model is instructed to generate:
- A new fused question 
- A set of hints to solve the question (not the solution)

After the fused questions and hints are generated. To evaluate coherence for training, two separate runs are performed on each generated question to generate solutions. 

- Generating solutions for the fused question using the provided hints
- And a separate run to generate solutions without using hints

This is a simple way to measure the self consistency of the model and correctness of the fused question along with the generated solution. 

---

# 3. Datasets used

- NuminaMath's  [CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) (Chain of thought) and [TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) (Tool Integrated Reasoning) splits were used 
- OpenMath's [CoT](https://huggingface.co/datasets/nvidia/OpenMathReasoning/viewer/default/cot) and [TIR](https://huggingface.co/datasets/nvidia/OpenMathReasoning/viewer/default/tir) splits. 
- [Putnam Archive](https://kskedlaya.org/putnam-archive/) questions from 1995 to 2025 were scraped

Details of the exact splits and filters applied on the samples used from these datasets are discussed below. 

---


# 4. Dataset splits

## 4.1 Fusion Datasets

### OpenMath Fusion Set

OpenMath dataset's CoT split was used, and based on the metadata tagged with problems which were in the ```category``` 
of  ``` "aops_c7_college_math" , "aops_c6_high_school_olympiads"  , "c7_college_math" , "aops_c4_high_school_math" ``` were used and had ```pass_rate_72b_tir``` to be less than ``` 0.4 ``` along with the final answer value to be in the range of ``` greater than 0, less than 100_000``` were chosen for fusion.

Around 8000 problems were filtered using this and used for fusion. Solutions were generated for these problems using [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)  which is later used for fusion process. Due to availability of integer values of the final solution, validation was easy and the filtered subset was classified into AA and AB type fusion pairs. 

### Putnam Fusion Set

Putnam archive's questions were also used for a separate fusion. Solutions were not regenerated like done for OpenMath's CoT fusion.

Why? 

This is because the solutions of questions from Putnam are not aligned with the format expected for AIMO (integer format). Some problems also were centered around mathematical theorems. Hence the ```AA``` and ```AB``` type classification is not done for now so no splits were done among the fusion pairs. They were treated neutral and assigned ```AA``` as there is no regeneration of solutions. Even after considering the fact that the Putnam bench might not always have integer format final solutions, but the difficulty of these problems are promising to enhance reasoning. Solutions for some fused questions (which would be released after the competition ends) via Putnam with multiple forward passes at 300k token limit per pass resulted in consistent answers. This was done on models stronger than GPT OSS showing the importance of such dataset.

## 4.2 Non Fusion Datasets

Apart from the fusion algorithm driven data, the following are also released. 
For the below mentioned data the solutions were generated using ```GPT OSS 120B``` in ```high``` reasoning with ```100k``` tokens allowed per individual question. 

### 4.2.1  OpenMath CoT split

Problems having  ```category``` of   ```aops_c7_college_math```, ```aops_c6_high_school_olympiads```  , ```c7_college_math``` , ```aops_c4_high_school_math``` were used and with ```pass_rate_72b_tir``` along with the final answer value to be in the range of ``` greater than 0, less than 100_000```

> NOTE: This was done for the fusion dataset mentioned above as well but the key difference is that in CoT split of OpenMath, there were tens of thousands of samples which did not have a value assigned to the ```pass_rate_72b_tir``` arg. For picking samples in the fusion set, samples which strictly had a value of ```pass_rate_72b_tir``` defined below ```0.4``` (40%) were chosen whereas for the non fusion, even if there was no value (usually tagged as ```n/a``` in the data) assigned to the ```pass_rate_72b_tir``` arg, it was still used. 

Solving mode: Pure CoT without any tool usage

### 4.2.2 OpenMath TIR split

Filtering is only done on problems whose solution had final answer in the 0 and 100_000 integer value. 

Solving mode: Tool integrated reasoning mode

### 4.2.3  NuminaMath CoT split

Problems having the ```source``` key in the metdata tag to be equal to ```olympiads``` are used. 

Solving mode: Tool integrated reasoning mode

### 4.2.4 NuminaMath TIR split

All available problems. 

Solving mode: Tool integrated reasoning mode


>NOTE:  For NuminaMath dataset there was no filtering done on the ```0 to 100_000 only``` rule. Many problems from the COT split had final answer value to be in fractions, mathematical expressions and so on. Discarding them would be a risk to lose important problems and hence this was not done. TIR split was relatively small with around 72k problems, due to this all problems were used. 



---

# 5. Dataset splits statistics


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F18445905%2Ff44247e4e9156e409126e1d3e8ef6a4a%2FScreenshot%202026-02-09%20124758.png?generation=1770621608241029&alt=media)



The activation aware fusion is a simple yet effective to smartly generate new math problems and aims to solve 2 main problems. Getting new data which the current open weight models might have not seen in their training data and to generate problems which have increased difficulty. The code setup used to plug the data in and to generate the fused questions, their hints and solutions is released in the below repo. The repo has code to do inference on the OpenMath and NuminaMath datasets mentioned above. The TIR code is not included in the repo as it is based on the highest scoring public notebook. The data is just swapped to use NuminaMath and the OpenMath data. Details regarding the dataset files, their structure and usage can be found in the ```README.md``` file found in the dataset repo.

```Code Repo: ```

Coming in a few days

```Kaggle Dataset Repo: ```

https://www.kaggle.com/datasets/aneeshmukkamala/aimo3afq

Please refer the ```README.md``` in the dataset link for detailed information on the files and their structure.
The dataset contains 235,998 unique problem instances, where each fused question is counted as one entry, even though both with-hints and without-hints model outputs are stored. This includes 177 Putnam and 36,473 OpenMath fused problems; counting the two outputs separately would give 272,648 generations

Information regarding experiments done on these datasets will be released soon.
Datapoints which exceeded the 100k character limit as per the guidelines would be released after the comeptition ends. 

