"""测试数据集：一批带人工标注的中文商品评论。

用途：
1. 当 few-shot 之外的小样本测试集
2. evaluate.py 用它反复跑，统计"格式正确率"（Pydantic 校验通过率）

注意：这里标注的 sentiment 是"参考答案"，用于人工核对抽取质量；
而 evaluate.py 的硬性验收指标是「格式正确率」——即输出能否通过 Pydantic 校验。
"""

SAMPLES = [
    {"text": "屏幕显示效果很棒，就是电池续航太差了，一天两充", "sentiment": "negative", "aspects": ["屏幕", "电池"]},
    {"text": "物流神速，昨天买今天到，包装也很严实", "sentiment": "positive", "aspects": ["物流", "包装"]},
    {"text": "一般般吧，没什么特别出彩的地方，也挑不出大毛病", "sentiment": "neutral", "aspects": []},
    {"text": "音质不错，低音很有弹性，但耳机戴着有点夹耳朵", "sentiment": "neutral", "aspects": ["音质", "佩戴"]},
    {"text": "客服态度极差，问问题半天不回，失望", "sentiment": "negative", "aspects": ["客服"]},
    {"text": "性价比很高，这个价位能买到这种配置，值了", "sentiment": "positive", "aspects": ["价格", "配置"]},
    {"text": "外观漂亮，手感也好，就是价格有点贵", "sentiment": "neutral", "aspects": ["外观", "手感", "价格"]},
    {"text": "用了一周就死机三次，质量太差了，不建议购买", "sentiment": "negative", "aspects": ["质量", "稳定性"]},
    {"text": "充电很快，半小时充满，发热控制得也不错", "sentiment": "positive", "aspects": ["充电", "发热"]},
    {"text": "颜色跟图片有色差，有点失望", "sentiment": "negative", "aspects": ["颜色"]},
]
