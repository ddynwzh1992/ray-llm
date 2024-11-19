

import os
import logging
import time
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
import logging

from llama_cpp import Llama



logger = logging.getLogger("ray.serve")

app = FastAPI()

# Define the deployment
@serve.deployment(name="LLamaCPPDeployment")
@serve.ingress(app)
class LLamaCPPDeployment:
    def __init__(self):
        # Initialize the LLamaCPP model
        self.model_id = os.getenv("MODEL_ID", default="SanctumAI/Llama-3.2-1B-Instruct-GGUF")
        # Get filename from environment variable with default fallback to "*Q4_0.gguf"
        self.filename = os.getenv("MODEL_FILENAME", default="*Q4_0.gguf")
        self.n_ctx = int(os.getenv("N_CTX"))
        # self.n_batch = int(os.getenv("N_BATCH"))
        # self.llama_cpp = Llama(model_path=MODEL_ID, n_ctx=self.n_ctx, n_batch=self.n_batch)
        self.llm = Llama.from_pretrained(repo_id=self.model_id,filename=self.filename,n_ctx=self.n_ctx)
        #"hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
        print("__init__ Complete")

    @app.post("/v1/chat/completions")
    async def call_llama(self, request: Request):
        try:
            body = await request.json()
            prompt = body.get("prompt")
            
            if not prompt:
                return JSONResponse(
                    status_code=400,
                    content={"error": "prompt is required"}
                )

            output = self.llm(
                "Q: " + prompt + " A: ",
                max_tokens=body.get("max_tokens", 32)
                # stop=body.get("stop", ["Q:", "\n"]),
                # echo=body.get("echo", True)
            )        
            
            return JSONResponse(content={
                "id": "cmpl-" + os.urandom(12).hex(),
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [{
                    "text": output["choices"][0]["text"],
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(output["choices"][0]["text"].split()),
                    "total_tokens": len(prompt.split()) + len(output["choices"][0]["text"].split())
                }
            })
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )




model = LLamaCPPDeployment.bind()
