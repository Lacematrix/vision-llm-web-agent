from scrapegraphai.graphs import SmartScraperGraph

OPENAI_API_KEY='sk-ac712e0af26440a48e21f3d9ec2a9a23'
OPENAI_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
OPENAI_MODEL='qwen3-vl-8b-instruct'

graph_config = {
    "llm": {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL,
    },
    "verbose": True,
    "headless": True,   # 根据是否需要渲染页面
}

smart = SmartScraperGraph(
    prompt="从该网页提取产品名称、价格、描述、是否有货",
    source="https://example-shop.com/product/12345",
    config=graph_config
)

result = smart.run()
print(result)
