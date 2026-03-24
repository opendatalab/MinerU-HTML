from mineru_html.base import MinerUHTMLCase, MinerUHTMLGenerateInput
from mineru_html.constants import ITEM_ID_ATTR
from mineru_html.exceptions import MinerUHTMLError, MinerUHTMLPromptError


def get_full_prompt_long_v0(html_str: str) -> str:
    try:
        prompt = f"""As a front-end engineering expert in HTML, your task is to analyze the given HTML structure and accurately classify elements with the {ITEM_ID_ATTR} attribute as either "main" (primary content) or "other" (supplementary content). Your goal is to precisely extract the primary content of the page, ensuring that only the most relevant information is labeled as "main" while excluding navigation, metadata, and other non-essential elements.
Guidelines for Classification:
Primary Content ("main")
Elements that constitute the core content of the page should be classified as "main". These typically include:
✅ For Articles, News, and Blogs:
The main text body of the article, blog post, or news content.
Images embedded within the main content that contribute to the article.
✅ For Forums & Discussion Threads:
The original post in the thread.
Replies and discussions that are part of the main conversation.
✅ For Q&A Websites:
The question itself posted by a user.
Answers to the question and replies to answers that contribute to the discussion.
✅ For Other Content-Based Pages:
Any rich text, paragraphs, or media that serve as the primary focus of the page.
Supplementary Content ("other")
Elements that do not contribute to the primary content but serve as navigation, metadata, or supporting information should be classified as "other". These include:
❌ Navigation & UI Elements:
Menus, sidebars, footers, breadcrumbs, and pagination links.
"Skip to content" links and accessibility-related text.
❌ Metadata & User Information:
Article titles, author names, timestamps, and view counts.
Like counts, vote counts, and other engagement metrics.
❌ Advertisements & Promotional Content:
Any section labeled as "Advertisement" or "Sponsored".
Social media sharing buttons, follow prompts, and external links.
❌ Related & Suggested Content:
"Read More", "Next Article", "Trending Topics", and similar sections.
Lists of related articles, tags, and additional recommendations.
Task Instructions:
You will be provided with a simplified HTML structure containing elements with an {ITEM_ID_ATTR} attribute. Your job is to analyze each element's function and determine whether it should be classified as "main" or "other".
Response Format:
Return a JSON object where each key is the {ITEM_ID_ATTR} value, and the corresponding value is either "main" or "other", as in the following example:
{{"1": "other","2": "main","3": "other"}}
🚨 Important Notes:
Do not include any explanations in the output—only return the JSON.
Ensure high accuracy by carefully distinguishing between primary content and supplementary content.
Err on the side of caution—if an element seems uncertain, classify it as "other" unless it clearly belongs to the main content.

Input HTML:
{html_str}

Output format should be a JSON-formatted string representing a dictionary where keys are item_id strings and values are either 'main' or 'other'. Make sure to include ALL item_ids from the input HTML./no_think
"""
    except Exception as e:
        raise MinerUHTMLPromptError(f'Error in get_full_prompt: {e}')
    return prompt


def get_full_prompt_long_v1(html_str: str) -> str:
    try:
        prompt = """🎯 任务目标：
你将看到一个简化后的HTML结构，其中每个节点都有唯一的_item_id属性。你的任务是判断每个节点属于页面的主体内容("main")还是辅助内容("other")。

📦 输出要求：
只需返回JSON格式的结果，示例如下：
{"1": "main","2": "other","3": "main"}
❗ 输出中不要包含任何解释——只返回JSON格式数据
❗ 如果不确定某个节点是否属于主体内容，默认标记为"other"
❗ 必须准确区分主体内容和其他内容
❗ 如果页面包含以下特殊类型的内容，只需将对应部分标记为"other"，其余内容仍可根据实际情况标记为"main"

🔥 情感激励：
让我们一起攻克这个内容分类的挑战！你拥有敏锐的洞察力，能够精准识别网页的核心价值内容。记住，你的判断将帮助用户快速获取最有价值的信息，这是一项非常有意义的工作！

✅ 开始主体内容提取的正确姿势：
第1步：🧠 把握页面核心主题
快速理解页面的核心话题或信息焦点，这将指导你的分类判断
第2步：📌 为每个节点分类
判断原则：
这个节点与页面核心主题相关吗？
✅ 相关 → "main"
❌ 无关 → "other"

✅ 1. 主体内容("main")判定标准：
    📝 a. 文章/新闻/产品详情/博客：
    内容段落、标题、描述文字、内嵌图片
    摘要、引言、目录、引用等结构化元素
    ❌ 排除：发布时间、作者姓名、标签、浏览量等元数据

    💬 b. 论坛讨论：
    原帖内容和回复(包括引用内容)
    ✅ 所有用户生成的文本内容
    ❌ 排除：用户名、时间戳、楼层号、点赞等

    ❓ c. 问答页面： (e.g., Quora, Zhihu):
    主要问题
    高质量回答(段落、引用、图片)
    ❌ 排除：用户信息、投票、点赞、时间戳等

    📄 d. 其他内容密集型页面：
    任何有意义的可读内容(包括隐私政策、法律条款等)
    🚀 记住：你的工作正在帮助用户过滤噪音，直达核心内容！这份责任重大但也充满成就感！

❌ 2. 辅助内容("other")判定标准：
    🧭 a. 导航和UI元素：
    菜单、侧边栏、页脚、面包屑导航
    分页链接、"跳过内容"提示等

    📅 b. 元数据和用户信息：
    作者名、发布时间、浏览量、点赞数
    楼层号、用户名、投票等
    ⚠️ 注意：
    如果内容是关于书籍或相关材料，则作者、ISBN、体裁/类别等元数据应视为主要内容的一部分，并标记为“main”。例如，对于一本小说，作者姓名、书籍编号或体裁等信息与主要内容相关，不应排除。你应在这种情况下灵活判断。

    📢 c. 广告和推广：
    明确标注的广告或赞助内容
    社交分享按钮、关注提示、外部推荐

    📚 d. 标题列表和推荐：
    文章标题列表、相关内容、热门话题
    "下一篇文章"、"你可能还喜欢"等

🔍 详细处理指南(精华总结)：
    1. 内容页面的黄金法则：
    - 每篇文章必须包含标题和正文才考虑"main"
    - 多篇文章要分别标注
    - 特殊情况：如果页面包含多篇不完整的文章，且每篇文章都以“阅读更多”之类的短语结尾，则整个页面应标记为“other”。
    ❌ 不要重复标注相同的元数据
    2. 特殊文档的处理：
    活动邀请、招聘信息、服务条款等有意义的内容都应标记为"main"
    目录、摘要、引言、引用等都是文章结构的一部分
    3. 元数据标注：
    ✅ 保留手工编写的联系信息（例如，职位发布中的联系信息、组织名称、电子邮件、电话号码、社交媒体账号）。
    ❌ 移除模板化的元数据（例如，每篇文章显示的相同作者、时间、点赞数）。
    4. 交互元素统一标准：
    所有"下载"、"阅读更多"、"点击"等按钮/链接都标记为"other"

💎 现在，展现你专业能力的时候到了！请分析以下HTML内容，并为每个_item_id返回分类结果：

Input HTML:
"""
        prompt = f'{prompt}{html_str}'
    except Exception as e:
        raise MinerUHTMLPromptError(f'Error in get_full_prompt_long_v1: {e}')
    return prompt


def get_full_prompt_long_v2(html_str: str) -> str:
    try:
        prompt = """🎯 角色与任务目标：
你是最精确的网页正文提取引擎。你的目标是作为无情的噪音过滤器，移除所有与页面核心主题不直接相关的辅助性内容和用户界面元素，以达到最高的正文召回率和精度。 专业约束： 你的判断必须基于HTML结构和内容相关性，而不是基于视觉呈现。
你将看到一个简化后的HTML结构，其中每个节点都有唯一的_item_id属性。你的任务是判断每个节点属于页面的主体内容("main")还是辅助内容("other")，保留网页正文部分，移除导航栏、元信息等无关部分。

📦 输出要求：
只需返回JSON格式的结果，示例如下：
{"1": "main","2": "other","3": "main"}
❗ 输出中不要包含任何解释——只返回JSON格式数据
❗ 如果不确定某个节点是否属于主体内容，默认标记为"other"
❗ 必须准确区分主体内容和其他内容

🔥 情感激励：
让我们一起攻克这个内容分类的挑战！你拥有敏锐的洞察力，能够精准识别网页的核心价值内容。记住，你的判断将帮助用户快速获取最有价值的信息，这是一项非常有意义的工作！

✅ 开始主体内容提取的正确姿势：
第1步：🧠 把握页面核心主题
快速理解页面的核心话题或信息焦点，这将指导你的分类判断
第2步：📌 为每个节点分类
判断原则：
这个节点与页面核心主题相关吗？
✅ 相关 → "main"
❌ 无关 → "other"

✅ 1. 主体内容("main")判定标准：
    📝 a. 文章/新闻/产品详情/博客/赛事信息：
    作者手动编辑撰写的内容，包括内容段落、标题、描述文字、内嵌图片、表格、代码块、多篇文章
    摘要、引言、目录、引用等结构化元素。
    ❌ 排除：辅助信息（如浏览量、点赞数、评论数、社交分享按钮）

    💬 b. 论坛讨论：
    原帖内容和回复(包括引用内容)
    ✅ 所有用户生成的文本内容(段落、引用、图片)
    ❌ 排除：用户名、时间戳、楼层号、点赞等

    📄 c. 其他内容密集型页面：
    任何有意义的可读内容(包括隐私政策、法律条款、网站说明等)
    ❌ 如网站声明、法律条款、服务协议、隐私政策等非核心主题内容，则标记为"other"（例如：出现在普通新闻页底部的服务条款链接）。
    ❌ 排除：没有元信息的纯链接模块，以及电商网站的无商品属性的商品列表
    🚀 记住：你的工作正在帮助用户过滤噪音，直达核心内容！这份责任重大但也充满成就感！

❌ 2. 辅助内容("other")判定标准：
    🧭 a. 导航和UI元素：
    菜单、侧边栏、页脚、面包屑导航
    分页链接、"跳过内容"提示等

    📅 b. 辅助性信息：
    任何描述内容载体状态或交互的辅助信息，应标记为"other"。
    - 属于此类的有：发布时间、作者名、浏览量、点赞数、评论数、楼层号、投票、、用户名等。
    - 例外（视为“main”）：任何直接定义或描述页面核心主题对象（文章、产品、书籍、服务、职位）的关键属性，应视为主要内容的一部分并标记为“main”。
        示例：书籍的作者/ISBN、商品的规格/价格、招聘的职位名称/薪资、活动邀请中的时间和地点。
        原则：如果移除这些信息会使页面核心主题不完整，则标记为"main"。

    📢 c. 广告和推广：
    文章中穿插的广告或赞助内容，与文章主题不相关
    社交分享按钮、关注提示、外部推荐

    📚 d. 推荐内容：
    相关内容、热门话题、推荐阅读、相关文章等
    "下一篇文章"、"你可能还喜欢"等

🔍 详细处理指南(精华总结)：
    1. 内容页面的黄金法则：
    - 网站正文有明确主题均可考虑"main"
    - 多篇文章要分别标注
    2. 特殊文档的处理：
    活动邀请、招聘信息、服务条款等有意义的内容都应标记为"main"
    目录、摘要、引言、引用等都是文章结构的一部分
    3. 辅助信息：
    保留（"main"）：手工编写的联系信息、产品规格、书籍作者/ISBN等直接定义核心主题的关键属性。
    移除（"other"）：模板化的、描述内容载体状态（如发布时间、浏览量）的辅助信息。
    4. 交互元素统一标准：
    所有"下载"、"阅读更多"、"点击"等按钮/链接都标记为"other"

💎 现在，展现你专业能力的时候到了！请分析以下HTML内容，并为每个_item_id返回分类结果：

Input HTML:
"""
        prompt = f'{prompt}{html_str}'
    except Exception as e:
        raise MinerUHTMLPromptError(f'Error in get_full_prompt_long_v1: {e}')
    return prompt


def get_full_prompt_long_compact(html_str: str) -> str:
    try:
        prompt = f"""As a front-end engineering expert in HTML, your task is to analyze the given HTML structure and accurately classify elements with the "_item_id" attribute as either "main" (primary content) or "other" (supplementary content). Your goal is to precisely extract the primary content of the page, ensuring that only the most relevant information is labeled as "main" while excluding navigation, metadata, and other non-essential elements.
Guidelines for Classification:
Primary Content ("main")
Elements that constitute the core content of the page should be classified as "main". These typically include:
✅ For Articles, News, and Blogs:
The main text body of the article, blog post, or news content.
Images embedded within the main content that contribute to the article.
✅ For Forums & Discussion Threads:
The original post in the thread.
Replies and discussions that are part of the main conversation.
✅ For Q&A Websites:
The question itself posted by a user.
Answers to the question and replies to answers that contribute to the discussion.
✅ For Other Content-Based Pages:
Any rich text, paragraphs, or media that serve as the primary focus of the page.
Supplementary Content ("other")
Elements that do not contribute to the primary content but serve as navigation, metadata, or supporting information should be classified as "other". These include:
❌ Navigation & UI Elements:
Menus, sidebars, footers, breadcrumbs, and pagination links.
"Skip to content" links and accessibility-related text.
❌ Metadata & User Information:
Article titles, author names, timestamps, and view counts.
Like counts, vote counts, and other engagement metrics.
❌ Advertisements & Promotional Content:
Any section labeled as "Advertisement" or "Sponsored".
Social media sharing buttons, follow prompts, and external links.
❌ Related & Suggested Content:
"Read More", "Next Article", "Trending Topics", and similar sections.
Lists of related articles, tags, and additional recommendations.
Task Instructions:
You will be provided with a simplified HTML structure containing elements with an "_item_id" attribute. Your job is to analyze each element's function and determine whether it should be classified as "main" or "other".
Response Format:
Return each item_id and its corresponding category in the following format:
{{item_id}}{{category}}{{item_id}}{{category}}...
Here, {{item_id}} may be 1, 2, 3..., and {{category}} is either 'main' or 'other', as shown in the example below:
"1main2other3other4main"
🚨 Important Notes:
Do not include any explanations in the output, return only the response in the format specified above.
Ensure high accuracy by carefully distinguishing between primary content and supplementary content.
Err on the side of caution—if an element seems uncertain, classify it as "other" unless it clearly belongs to the main content.
The output format should only contain item_id numbers and categories (either "main" or "other"), without any additional characters. Make sure to include ALL item_ids from the input HTML.
Input HTML:
{html_str}"""
    except Exception as e:
        raise MinerUHTMLPromptError(f'Error in get_full_prompt_long_compact: {e}')
    return prompt


def get_full_prompt_short_compact(html_str: str) -> str:
    try:
        prompt = f"""As an HTML expert, classify elements with "_item_id" as "main" or "other", keeping only the main content and removing nav, metadata, etc.
Guidelines:
"Main": Includes primary content like article text, images in the article, original posts in forums, Q&A questions, and answers.
"Other": Includes navigation, metadata, ads, user info, and related content (e.g., sidebars, timestamps, suggested articles).
Output Format:
Return each _item_id and its corresponding category in the following format:
{{_item_id}}{{category}}{{_item_id}}{{category}}...
Here, {{_item_id}} may be 1, 2, 3..., and {{category}} is either "main" or "other", as shown in the example below:
"1main2other3other4main"
Input HTML:
{html_str}"""
    except Exception as e:
        raise MinerUHTMLPromptError(f'Error in get_full_prompt_short_compact: {e}')
    return prompt


def get_full_prompt(html_str: str, version: str = 'v0') -> str:
    if version == 'v0':
        return get_full_prompt_long_v0(html_str)
    elif version == 'v1':
        return get_full_prompt_long_v1(html_str)
    elif version == 'v2':
        return get_full_prompt_long_v2(html_str)
    elif version == 'compact':
        return get_full_prompt_long_compact(html_str)
    elif version == 'short_compact':
        return get_full_prompt_short_compact(html_str)
    else:
        raise MinerUHTMLPromptError(f'Unsupported prompt version: {version}')


def build_prompt(input_case: MinerUHTMLCase, prompt_version: str) -> MinerUHTMLCase:
    """
    Build prompt for model inference

    Args:
        process_data: process data
        prompt_version: prompt version
    Returns:
        generate input
    """
    try:
        prompt = get_full_prompt(input_case.process_data.simpled_html, prompt_version)
        input_case.generate_input = MinerUHTMLGenerateInput(full_prompt=prompt)
        return input_case
    except Exception as e:
        if isinstance(e, MinerUHTMLError):
            e.set_case_id(input_case.case_id)
            raise e
        else:
            raise MinerUHTMLPromptError(
                f'Build prompt failed: {str(e)}', case_id=input_case.case_id
            ) from e
