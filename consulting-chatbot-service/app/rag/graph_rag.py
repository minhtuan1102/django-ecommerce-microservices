import os
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain

class GraphRAGService:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
        self.username = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.graph = Neo4jGraph(url=self.uri, username=self.username, password=self.password)
        self.llm = self._build_llm()
        self.chain = self._build_chain()

    def _build_llm(self):
        llm_type = os.getenv("LLM_TYPE", "gemini")
        if llm_type == "gemini":
            return ChatGoogleGenerativeAI(
                model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0
            )
        else:
            return ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0
            )

    def _build_chain(self):
        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True
        )

    def query(self, question: str):
        try:
            result = self.chain.invoke({"query": question})
            return result.get("result") or result.get("answer")
        except Exception as e:
            return f"Error querying KB_Graph: {str(e)}"
