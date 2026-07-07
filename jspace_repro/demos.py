# Demo prompt set for the j-space visualization. Each demo reproduces one
# phenomenon from the paper; ranks quoted in the notes were measured on
# Qwen3.5-0.8B with the Neuronpedia wikitext lens (probe scripts in
# experiments/).
from dataclasses import dataclass, field

from jlens.examples import Example


@dataclass(frozen=True)
class Demo:
    slug: str
    example: Example
    title_en: str
    title_zh: str
    desc_en: str
    desc_zh: str
    # Words to always rank-track (only those that map to single tokens are kept).
    concepts: list[str] = field(default_factory=list)
    # Cell auto-selected when the page loads: (position from end, lens layer).
    focus_pos: int = -1
    focus_layer: int = 12
    note_en: str = ""
    note_zh: str = ""


DEMOS: list[Demo] = [
    Demo(
        slug="multihop-boot",
        example=Example(
            slug="multihop-boot",
            section="Multi-hop reasoning",
            description="",
            user=(
                "In the country shaped like a boot, what currency do they use? "
                "Answer with only the currency name."
            ),
        ),
        title_en="Multi-hop: boot → Italy → currency",
        title_zh="多跳推理:靴子 → 意大利 → 货币",
        desc_en=(
            "A two-hop question. The intermediate answer 'Italy' never appears "
            "in the prompt or the output — but the J-lens reads it out in the "
            "middle layers."
        ),
        desc_zh=(
            "两跳问题。中间答案 Italy 从未出现在输入或输出里——但 J-lens 在中间层"
            "直接读出了它。"
        ),
        concepts=[" Italy", "Italy", " Italian", " euro", " Euro", " lira", "Dollar"],
        focus_pos=-1,
        focus_layer=11,
        note_en=(
            "At the final position, ' Italy' climbs to #8 of 151,936 tokens around layer 11 "
            "— the silent first hop. The 0.8B model then fumbles the second "
            "hop in its spoken answer, so the thought is visible even though "
            "the output is wrong."
        ),
        note_zh=(
            "在最后一个位置,' Italy' 在第 11 层附近升至全词表第 8 名 (#8)——沉默的第一跳。"
            "0.8B 模型随后在第二跳上失误,答案是错的,但想法本身清晰可见。"
        ),
    ),
    Demo(
        slug="capital-cn",
        example=Example(
            slug="capital-cn",
            section="Thought vs speech",
            description="",
            user="形状像靴子的国家的首都是哪座城市?只回答城市名。",
        ),
        title_en="Thinking 'Italy', saying 'Moscow'",
        title_zh="心里想着「意大利」,嘴上说出「莫斯科」",
        desc_en=(
            "Chinese two-hop question (capital of the boot-shaped country). "
            "Internally the English token ' Italy' reaches #2 of the whole vocabulary — yet the "
            "model answers 莫斯科 (Moscow). Workspace and output dissociate, "
            "and the silent thought is in a different language than the "
            "conversation."
        ),
        desc_zh=(
            "中文两跳问题(靴子形状国家的首都)。模型内部,英文 token ' Italy' "
            "升至全词表第 2 名——但嘴上回答 莫斯科。工作空间与输出解离,而且沉默的想法"
            "用的是与对话不同的语言。"
        ),
        concepts=["意大利", "罗马", "莫斯科", " Italy", " Rome"],
        focus_pos=-1,
        focus_layer=12,
        note_en=(
            "' Italy' hits #2 at layer 12 — the first hop succeeds, and "
            "the model thinks it in English despite the Chinese prompt. The "
            "correct answer 罗马/' Rome' peaks around #83 at layer 13 and "
            "fades; 莫斯科 (Moscow) wins the output layers instead. A two-hop "
            "failure you can watch happen."
        ),
        note_zh=(
            "' Italy' 在第 12 层达到第 2 名 (#2)——第一跳成功,而且尽管问题是中文,"
            "模型是用英文 token 想的。正确答案 罗马/' Rome' 在第 13 层最好到"
            "第 83 名便消退;莫斯科 反而赢得了输出层。你可以亲眼看着第二跳失败。"
        ),
    ),
    Demo(
        slug="sport",
        example=Example(
            slug="sport",
            section="Category naming",
            description="",
            user="Think of a sport. Just name it, one word.",
        ),
        title_en="Category → instance: think of a sport",
        title_zh="类别 → 实例:想一种运动",
        desc_en=(
            "The paper's category-naming protocol. Before any word is emitted, "
            "candidate sports light up in the workspace and compete."
        ),
        desc_zh=(
            "论文的类别命名协议。在输出任何词之前,候选运动已在工作空间中点亮并"
            "相互竞争。"
        ),
        concepts=[" Football", " football", " soccer", " basketball", " tennis",
                  " Soccer", " Basketball"],
        focus_pos=-1,
        focus_layer=20,
        note_en=(
            "' Football' climbs to #2 by layer 20 at the final position, and wins #1 in the output layer — "
            "and this time the model says it. Watch rival sports trail just "
            "behind in the top-k list."
        ),
        note_zh=(
            "' Football' 在第 20 层升至第 2 名,并在输出层夺得第 1 名——这一次模型也确实说出了它。"
            "注意 top-k 列表里紧随其后的其他候选运动。"
        ),
    ),
    Demo(
        slug="mod-topic",
        example=Example(
            slug="mod-topic",
            section="Voluntary modulation",
            description="",
            user=(
                'Write "She carefully placed the letter back inside the wooden '
                'drawer." Concentrate on ocean creatures while you write the '
                "sentence. Don't write anything else."
            ),
            assistant_prefill=(
                "She carefully placed the letter back inside the wooden drawer."
            ),
        ),
        title_en="Voluntary modulation: think of the ocean",
        title_zh="自主调制:一边写字一边想海洋",
        desc_en=(
            "Paper protocol: write a fixed sentence while concentrating on an "
            "unrelated topic. The written words are about a drawer; the "
            "workspace hums with ocean creatures."
        ),
        desc_zh=(
            "论文协议:写一个固定句子,同时专注于一个无关主题。写出来的字关于"
            "抽屉;工作空间里却满是海洋生物。"
        ),
        concepts=[" ocean", " Ocean", " fish", " whale", " sea", " shark",
                  " dolphin", " marine"],
        focus_pos=-1,
        focus_layer=19,
        note_en=(
            "At the sentence-final period, an ocean word is the lens's #1 "
            "readout at layer 19 — deliberate, instruction-driven control of "
            "workspace contents, exactly as in the paper."
        ),
        note_zh=(
            "在句末句号处,海洋词在第 19 层是 lens 读出的第 1 名 (#1)——"
            "这是指令驱动的、对工作空间内容的自主控制,与论文一致。"
        ),
    ),
    Demo(
        slug="mod-arith",
        example=Example(
            slug="mod-arith",
            section="Voluntary modulation (arithmetic)",
            description="",
            user=(
                'Write "The old painting hung crookedly on the wall." '
                "Try to focus on evaluating 3^2 - 2 while you write the "
                "sentence. Don't write anything else."
            ),
            assistant_prefill="The old painting hung crookedly on the wall.",
        ),
        title_en="Voluntary modulation: mental arithmetic (partial)",
        title_zh="自主调制:心算(部分复现)",
        desc_en=(
            "Same protocol with mental math (3² − 2 = 7) as the side task. On "
            "this 0.8B model the effect is weak — an honest partial result "
            "showing workspace capacity limits at small scale."
        ),
        desc_zh=(
            "同一协议,副任务换成心算(3² − 2 = 7)。在 0.8B 模型上效果较弱——"
            "一个诚实的部分复现,体现小模型工作空间容量的局限。"
        ),
        concepts=["7", " 7", "9", " 9", " seven", " nine"],
        focus_pos=-1,
        focus_layer=14,
        note_en=(
            "The digit '7' only reaches ≈#58 (layer 14) during the "
            "sentence — present but faint. The paper ran this on much larger "
            "models; scale matters for what the workspace can hold."
        ),
        note_zh=(
            "数字 '7' 在写句子期间最好只到约第 58 名(第 14 层)——存在但微弱。"
            "论文用的是大得多的模型;工作空间能装下什么,与规模有关。"
        ),
    ),
    Demo(
        slug="fear",
        example=Example(
            slug="fear",
            section="Don't say it",
            description="",
            user="Think about your greatest fear, but don't say it.",
        ),
        title_en="Think it, don't say it",
        title_zh="想它,但别说出来",
        desc_en=(
            "The model is asked to hold a thought while suppressing it from "
            "output. The lens reads fear-related concepts that never surface."
        ),
        desc_zh=(
            "要求模型持有一个想法但抑制输出。lens 读出了从未浮出水面的恐惧相关"
            "概念。"
        ),
        concepts=[" fear", " death", " dying", " dark", " darkness", " alone",
                  " failure", " Fear"],
        focus_pos=-1,
        focus_layer=17,
        note_en=(
            "Fear-related words reach ≈#17 at layer 17 while the actual "
            "output stays evasive — held-but-unspoken content."
        ),
        note_zh=(
            "恐惧相关词在第 17 层达到约第 17 名,而实际输出保持回避——被持有"
            "但未说出的内容。"
        ),
    ),
]
