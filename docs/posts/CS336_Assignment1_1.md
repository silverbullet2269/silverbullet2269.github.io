---
title: CS336 Assignment1 (Part 1)
date: 2026-11-11
description: CS336 Assignment1 学习笔记
---

# Byte-Pair Encoding (BPE) Tokenizer
## The Unicode Standard
Unicode 是将字符映射为整数码点 (integer code point) 的文本编码标准。

`ord(): character -> integer code point`

`chr(): integer code point -> character`

如图所示，汉字 “牛” 对应的 unicode 编码为 29275，一般记为 U+725B，U+为常规前缀，725B 为 29275 的十六进制表示。
![](\images\CS336_Assignment1\img1.png)

## Unicode Encodings
使用 Unicode code points 来直接训练 tokenizers 是不切实际的，因为词汇量会很大（约 150k 个物品）而且非常稀疏（类似生僻字，基本不出现）。所以使用 unicode encoding，将 unicode character 进一步转换为一串字节。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32。
```
encode("utf-8"): unicode string -> UTF-8 byte string
list(): get the byte values (integers from 0 to 255)
## 0-255是对于utf-8来说，2^8=256
decode("utf-8"): UTF-8 byte string -> unicode string
```

如下图所示，初始字符为 "hello! 我是一名研究生"，通过编码将其转换为 bytes 类型。其中里面：hello! 还能正常看到。这是因为 ASCII 字符在 UTF-8 中只需要一个 byte，而且 byte 值刚好和 ASCII 编码一致。而中文编码后需要 3 bytes，所以 “我” 编码得到 “\xe6\x88\x91”，对应的字节值为 [230, 136, 145]，其余类似。

![](\images\CS336_Assignment1\img2.png)

![](\images\CS336_Assignment1\img3.png)

![](\images\CS336_Assignment1\img4.png)

## Subword Tokenization
Byte-level tokenization 解决了 word-level tokenizer 容易遇到的词表外（out-of-vocabulary）问题，因为任何文本都可以被表示成 UTF-8 字节序列。然而，它也带来了输入序列过长的问题。例如，一个包含 10 个单词的句子，在 word-level tokenizer 中可能只需要约 10 个 token，而使用 byte-level tokenizer 时可能需要 50 个甚至更多 token，具体取决于文本的字节长度。更长的序列不仅会增加模型的计算开销，还会在数据中形成更长程的依赖（long-term dependencies）：模型往往需要跨越更多 byte token，才能组合出完整的字符、子词、单词乃至更高层次的语言结构，因此语言建模会更加困难。
Subword tokenization 是 word-level 与 byte-level tokenization 之间的一种折中方案。它仍然保留 byte token 作为最基本的表示单位，但会把训练语料中频繁共同出现的 token 序列逐渐合并成更大的 token。例如，原本 "t"、"h"、"e" 是三个独立的 byte token，如果它们经常连续出现，BPE 可能先将 "t" 和 "h" 合并为 "th"，再进一步将 "th" 和 "e" 合并为 "the"。之后 "the" 就可以用一个 token 表示，从而缩短输入序列。BPE（Byte-Pair Encoding）就是通过不断合并当前最频繁的相邻 token 对，逐步构建这些 subword token 和最终词表的方法
## BPE Tokenizer Training
BPE tokenizer训练过程主要包括三个步骤：
### Vocabulary initialization
词汇表是 bytestring token 与整数 ID 之间的一对一映射关系，所以 byte-level BPE tokenizer 的初始词汇表是所有字节的集合，1 byte = 8 bits，有 256 种可能，所以初始词汇表的大小为 256。例如：
```
b"A" → 65
b"B" → 66
```
### Pre-tokenization
理论上可以在整个语料库的 byte 序列上统计相邻字节出现的频率并对频率最高的字节进行合并，为什么需要 pre-tokenization？
- 计算成本太高：每次合并都需要扫描一次语料库。
- 不合理合并：直接合并可能导致产生仅在标点符号上有区别的 token，例如 dog 与 dogs!。

因此，我们对语料库进行预分词 (pre-tokenize)，即先进行粗粒度分词，这有助于我们统计字符对出现的频率。因为训练的是 byte-level BPE model，所以每个 pre-token 都用 UTF-8 字节序列表示。

例如，"text" 是一个 pre-token，出现了 10 次，那么我们在统计 't' 和 'e' 相邻出现的频率时，可以直接加 10，而不需要遍历语料库。
原始的 BPE 通过空格来进行预分词 (i.e., s.split(" "))。而现代分词器大多采用基于正则表达式的预分词器 (例如 GPT2)，如下：

`PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""`

![](\images\CS336_Assignment1\img5.png)

具体例子如下图所示，不过，在代码中使用时，你应该使用 re.finditer，因为 findall 会将所有预分词得到的匹配结果构建 list 全部储存在内存中，但 finditer 是逐个产生 pre-token 并统计频率。

![](\images\CS336_Assignment1\img6.png)

### Compute BPE merges

1. 统计每个 pre-token 内部所有相邻 token pair 的频率
2. 找到出现频率最高的 pair，如 (A, B)
3. 将所有 (A, B) 合并为新 token AB
4. 将 AB 加入 vocabulary
5. 重复上述过程，直到达到目标词表大小

初始词表大小为 256，所以最终 vocabulary size = 256 + merge 次数。
注意为了提高 BPE 训练的效率，不考虑跨 pre-token 边界的配对，例如两个 pre-token : some | text，在进行 merge 时，不可以合并 'e' 和 't'，因为这样就跨过了 pre-token 边界，只能在 some 和 text 内部分别进行 merge。
在计算合并时，若成对频率出现平局，应确定性地优先选择字典序更大的配对来打破平局。例如，如果配对 (“A”, “B”)、(“A”, “C”)、(“B”, “ZZ”) 和 (“BA”, “A”) 的频率均最高，则应合并 (“BA”, “A”)。

![](\images\CS336_Assignment1\img7.png)

### Special tokens
special token 用于表示元信息，例如：
```
<|endoftext|>可以表示文档或序列结束。
```
special token 必须始终作为一个完整 token，不能被 BPE 拆分。

special token 需要直接加入 vocabulary，并拥有固定 token ID。

special token 不是通过 BPE frequency merge 学出来的，而是人为指定的。
### Example : BPE training example

在后面作业中要求 vocab 类型为 dict[int, bytes]，这里按照自己思路设为了 dict[bytes, int]，但总体思路不变。

![](\images\CS336_Assignment1\img8.png)

![](\images\CS336_Assignment1\img9.png)

![](\images\CS336_Assignment1\img10.png)
## Experimenting with BPE Tokenizer Training
目标：在 TinyStories 的数据集上训练一个字节级的 BPE 分词器。
### Parallelizing pre-tokenization
例子：file_size=800 bytes，desired_num_chunks=4

![](\images\CS336_Assignment1\img11.png)

![](\images\CS336_Assignment1\img12.png)

### Removing special tokens before pre-tokenization
在 BPE训练前进行预分词时，必须先将语料按 special token分割成独立文档，然后分别对每个文档进行预分词，这样做的目的是防止不同文档的文本在文档边界处发生跨文档合并，例如语料 `[Doc 1]<|endoftext|>[Doc 2]` 必须分割为 `[Doc 1]` 和 `[Doc 2]` 分别处理，确保 `Doc 1` 末尾的字符永远不会与 `Doc 2` 开头的字符被误认为相邻而统计到一起；同时 special token 本身（如 `<|endoftext|>`）应从文本中移除，不参与词频统计，因为它们只是控制分隔符而非真正的语言内容。实现时需使用 `re.split` 配合 `re.escape` 构建分割模式，例如 `pattern = "|".join(re.escape(tok) for tok in special_tokens)`，这样可以正确处理包含 `|` 等正则特殊字符的令牌。最后，`test_train_bpe_special_tokens` 测试将验证你是否正确实现了这一逻辑。

解释 `pattern = "|".join(re.escape(tok) for tok in special_tokens)`：
- re.escape 是将 special token 中的特殊符号进行转义，例如 `<|endoftext|>`中的 `'|'`，然后 "|".join() 是将转义后的 special tokens 使用或 ('|') 连接起来。
### Optimizing the merging step
上面示例中朴素的 BPE 训练实现之所以很慢，是因为在每次合并时，它都会遍历所有的字节对，以找出频率最高的那一对。然而，每次合并后，只有与被合并的字节对重叠的那些字节对的计数会发生变化。因此，可以通过索引所有字节对的计数并增量更新这些计数来提高 BPE 训练速度，而不是显式地遍历每个字节对来计算频率。通过这种缓存机制，你可以获得显著的性能提升，尽管我们需要注意的是，BPE 训练的合并部分在 Python 中无法并行化。


> [!NOTE] Low-Resource Tips
> - 性能分析 (Profiling)：使用 profiling 工具（例如 cProfile 或 py-spy）精确定位耗时最多的代码，集中精力优化真正的瓶颈
> - ”降级“ (Downscaling)：避免直接使用完整数据集（如TinyStories的212万篇文档），而是先在经过精心挑选的小规模调试集（如仅包含2.2万篇文档的验证集）上运行，确保该调试集既能暴露与完整配置相同的性能瓶颈（如词频统计或合并更新），又能将单次运行时间控制在短时间内以支持快速迭代，待逻辑与性能验证无误后再切换至完整数据进行最终训练。

### Code
![](\images\CS336_Assignment1\img13.png)

在开始之前，先再次强调代码中涉及的数据类型的变化

![](\images\CS336_Assignment1\img14.png)

代码在 assignment1-basics\cs336_basics\bpe_train_v1.py 和 assignment1-basics\cs336_basics\bpe_train_v2.py

其中 bpe_train_v1 未通过速度测试，所以改进得到 bpe_train_v2，此时可通过速度测试。

![](\images\CS336_Assignment1\img15.png)

![](\images\CS336_Assignment1\img16.png)

下面分析所作的改进之处：

首先 v1：每次都创建新对象 bytes([b]) 。每次调用都分配内存，对于 13,000 字符，就需要创建 13,000 个 bytes 对象。

v2：直接索引预创建对象 BYTE_TOKENS[b] 。O(1) 查找，不分配新内存，最开始 256 个 bytes 对象，重复使用。

![](\images\CS336_Assignment1\img17.png)

进行测试，如下，确实可节省一部分时间，且数据量越大，节省时间会越多。

![](\images\CS336_Assignment1\img18.png)

![](\images\CS336_Assignment1\img19.png)

v2 的 `pre_tokenize` 相比 v1，主要提升了壮性、正确性、效率和可读性。首先，它对 `special_tokens` 为空的情况进行了单独处理，避免 `pattern=""` 时执行 `re.split("", text)`，例如 `re.split("", "hello")` 会得到 `["", "h", "e", "l", "l", "o", ""]`，导致文本被错误地按字符切分；改进后会直接把整个 `"hello"` 作为一个 chunk 继续处理。其次，它将 special token 按长度从长到短排序，可以避免前缀冲突，例如同时存在 `"<|end|>"` 和 `"<|end|><|start|>"` 时，如果短 token 先参与匹配，就可能提前匹配 `"<|end|>"`，从而破坏较长 special token 的整体识别，而最长优先可以避免这个问题。另外，v2 使用 `Counter` 直接进行词频统计，并把 token 转换结果保存下来，只转换一次；v1 处理 `"hello"` 时可能先调用一次 `str_to_tuple("hello")` 判断是否存在，再调用一次进行计数，而新版只需先得到 `token`，再执行 `token_counts[token] += 1`。同时，v2 显式先将字符串编码为 UTF-8 bytes，再转换为 tuple，使 BPE 基于字节处理的逻辑更加清晰。

![](\images\CS336_Assignment1\img21.png)

下面也做一个实验测试，由于 str_to_tuple 的问题上面已验证过，所以这里测试时也将 v1 的修改为与 v2 一致，只用于验证 Counter() + 只一次 str_to_tuple 是否节约时间。
根据结果可得到避免重复 str_to_tuple() 确实可以节约时间，而 Counter 的主要收益是可读性和代码简洁性。甚至 Counter 所用时间会略长一些。

![](\images\CS336_Assignment1\img20.png)

![](\images\CS336_Assignment1\img22.png)

最后，是最重要的优化。v2 版本 将 v1 版本“每轮 merge 都重新扫描全部 pre-token、重新统计所有 pair”的全量计算方式，改成了增量更新。程序首先只完整计算一次 pair_counts，同时通过 pair_to_words 建立 pair → word_id 的倒排索引；之后选择出 best_pair 时，只处理真正包含该 pair 的 word，并通过“减去旧 word 的 pair 贡献、加入新 word 的 pair 贡献”来维护全局统计，整体过程是：

1. 通过 pair_to_words[best_pair] 找到所有包含它的 word_id，即 affected_words。
2. 对每个受影响的 word，把它旧有的所有相邻 pair 从 pair_counts、pair_to_words 中移除。
3. 在该 word 内合并 best_pair，得到新的 word。
4. 重新统计新 word 的所有相邻 pair。
5. 把新统计结果重新加入 pair_counts 和 pair_to_words。

因此避免了每轮重复扫描整个语料。

这里直接给出实践对比结果：

![](\images\CS336_Assignment1\img23.png)

![](\images\CS336_Assignment1\img24.png)

TinyStories 数据集下 bpe_train_v2 运行结果（assignment1-basics\cs336_basics\train_bpe_tinystories.py）如下，可以看到运行时间约为 35 mins，大于 30 mins。

![](\images\CS336_Assignment1\img25.png)

由于 Hints 中提及并行可令时间降低至 2 mins，所以编写 bpe_train_v3，在 pre_tokenize 处修改为并行操作。在主进程中通过 find_chunk_boundaries 函数 (assignment1-basics\cs336_basics\pretokenization_example.py) 找到文件 boundaries，然后把 path/start/end 发给每个子进程 worker。每个 worker 打开文件，读取文件区间进行 pre-tokenize，最后返回 Counter。

TinyStories 数据集下 bpe_train_v3 运行结果（assignment1-basics\cs336_basics\train_bpe_tinystories.py）如下，可以看到运行时间约为 4 mins，但仍大于 2 mins。

![](\images\CS336_Assignment1\img26.png)

bpe_train_v4：优化 pre-tokenization：先统计字符串，再把 unique pre-token 转成 bytes tuple，不要对每一次出现都重复 `encode + bytes_to_tuple`。

例如文本分词后得到：
```text
the, cat, the, cat, the
```
v3 版本会对每次出现都执行：
```text
"the" -> encode -> bytes_to_tuple   # 3 次
"cat" -> encode -> bytes_to_tuple   # 2 次
```
总共转换 **5 次**。
v4 版本先统计：
```python
string_counts = {
    "the": 3,
    "cat": 2,
}
```
再转换：
```text
"the" -> encode -> bytes_to_tuple   # 1 次，count=3
"cat" -> encode -> bytes_to_tuple   # 1 次，count=2
```
总共只转换 **2 次**。
因此，如果：
* `N` = pre-token 总出现次数
* `U` = unique pre-token 数量

那么编码和 `bytes_to_tuple` 的调用次数从：
```text
N 次 -> U 次
```
自然语言中通常 `U << N`，所以能减少大量重复计算。

![](\images\CS336_Assignment1\img27.png)

bpe_train_v5：继续改进，v4 每轮选择 `best_pair` 时，都需要遍历整个 `pair_counts` 执行 `max()`；现在改用 max-heap 后，可以直接从堆顶获取当前最高频 pair，避免每轮进行全量扫描。  
同时，在 merge 更新阶段，v4 虽然已经优化为只处理包含目标 pair 的相关 words，但仍可能重新统计这些 words 中的所有 pair；现在进一步利用 `word_pair_counts` 做增量更新，只调整 merge 前后实际发生变化的 pair 的全局计数，而 word 中未发生变化的 pair 不需要重复更新。

![](\images\CS336_Assignment1\img28.png)

## BPE Tokenizer: Encoding and Decoding

![](\images\CS336_Assignment1\img29.png)

代码：assignment1-basics\cs336_basics\tokenizer.py

python -X utf8 -m pytest tests/test_tokenizer.py -vv

![](\images\CS336_Assignment1\img30.png)

-X utf8：它强制 Python 使用 UTF-8 模式，避免某些 Windows 环境下默认编码不是 UTF-8 而导致的问题。
在 PowerShell 或 CMD 中执行：`setx PYTHONUTF8 1`
现在可以直接运行 python -m pytest tests/test_tokenizer.py

![](\images\CS336_Assignment1\img31.png)