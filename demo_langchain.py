import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 0. 配置你的大模型 API 密钥 (这里以 OpenAI 为例)
# 在实际运行前，请在终端设置环境变量：export OPENAI_API_KEY="sk-xxxx"

load_dotenv()  # 从 .env 文件加载环境变量
client = OpenAI()  # 初始化 OpenAI 客户端

# 1. 实例化大模型 (大脑)
# 选择一个支持 Tool Calling 的模型，比如 gpt-4o 或 gpt-3.5-turbo
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

my_api_key = os.getenv("DEEPSEEK_API_KEY")
if not my_api_key:
    raise ValueError("🚨 致命错误：找不到 API Key！请检查 .env 文件是否配置了 DEEPSEEK_API_KEY。")

llm = ChatOpenAI(
    api_key=my_api_key,
    base_url="https://api.deepseek.com",  # 核心魔法：把请求发给 DeepSeek 而不是 OpenAI
    model="deepseek-chat",                # 替换模型名称
    temperature=1.0
)

# ==========================================
# 2. 定义工具 (Agent 的手和脚)
# ==========================================

# 工具 A：搜索新闻 (使用现成的 DuckDuckGo 工具)
search_tool = DuckDuckGoSearchRun(
    name="search_stock_news",
    description="用于搜索当天的实时股票新闻和市场动态。"
)

# 工具 B：查询数据库获取历史股价 (使用 @tool 装饰器自定义工具)
@tool
def get_historical_prices(ticker: str) -> str:
    """
    用于查询数据库中某只股票最近5天的历史股价。
    参数 ticker 必须是股票代码，例如 'AAPL' 或 'TSLA'。
    """
    print(f"\n[系统提示: Agent 正在调用数据库查询 {ticker} 的历史价格...]\n")
    
    # 这里模拟一个本地数据库查询过程（实际业务中这里会写 SQL 查询语句）
    mock_db = {
        "AAPL": "2026-03-04: $170 | 2026-03-05: $172 | 2026-03-06: $171 | 2026-03-07: $175 | 2026-03-08: $174",
        "TSLA": "2026-03-04: $180 | 2026-03-05: $182 | 2026-03-06: $178 | 2026-03-07: $179 | 2026-03-08: $185",
    }
    
    # 返回查询结果给大模型
    return mock_db.get(ticker.upper(), f"数据库中未找到 {ticker} 的历史数据。")

# 把所有工具打包成一个列表
tools =[search_tool, get_historical_prices]

# ==========================================
# 3. 设计 Prompt (给 Agent 下达 SOP 工作规范)
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名华尔街资深股票分析师。
    你的任务是根据用户提供的股票代码，写一份专业的简短分析报告。
    
    你必须严格按照以下步骤执行：
    第一步：使用 'search_stock_news' 工具搜索该公司的最新新闻和市场情绪。
    第二步：使用 'get_historical_prices' 工具获取该股票最近的历史价格趋势。
    第三步：结合新闻和历史价格，写出一份包含【最新动态】、【价格趋势】和【分析师建议】的中文研报。
    """),
    ("human", "{input}"),
    # agent_scratchpad 是必需的，用于记录 Agent 每一步“思考-调用工具-得到结果”的中间过程
    ("placeholder", "{agent_scratchpad}"),
])

# ==========================================
# 4. 组装 Agent 并运行
# ==========================================

# 创建工具调用 Agent
#agent = create_tool_calling_agent(llm, tools, prompt)
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 Agent 调度器 (AgentExecutor 负责真正地执行循环：思考 -> 调工具 -> 思考 -> 输出)
# verbose=True 会在终端打印出 Agent 思考的完整过程，非常炫酷！
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 触发任务！
if __name__ == "__main__":
    #print("🚀 Agent 启动，开始执行任务...\n")
    print("[Start] Agent 启动，开始执行任务...\n")
    
    # 给 Agent 下达用户指令
    user_input = "帮我写一份今天关于苹果公司（AAPL）的股票分析报告。"
    
    # 运行 Agent
    response = agent_executor.invoke({"input": user_input})
    
    print("\n" + "="*50)
    print("最终分析报告：")
    print("="*50)
    print(response["output"])