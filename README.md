# llama-ecosystem-prompting

This repository is a modular workshop that consists of various hands-on labs that walk through deploying llama 2 models on Amazon Bedrock / Amazon SageMaker JumpStart to use advanced prompt engineering techniques and perform various prompt engineering evaluations. These notebooks are meant be to customizable for developers to clone and modularize the evaluation labs or benchmark their own prompt engineering techniques on the other foundational models provided by Amazon Bedrock and Amazon SageMaker JumpStart.

## Modularizing each lab

This repo allows you to go through each lab to learn about advanced prompt engineering (including using LangChain agents on Amazon Bedrock), prompt compression, and how to evaluate llm's using promptbench and Purple Llama's CyberSecEval.

However, the purpose of each notebook is to allow GenAI engineers and developers the building blocks to quickly deploy their own llm on Amazon Bedrock / SageMaker JumpStart and perform tailorized prompt engineering techniques / evaluations with the promptbench evaluation library. With each notebook having customizable cells, this allows developers to deploy various Foundational Model's provided on Amazon Bedrock outside of the Llama 2 model's used in the labs.

### Llama2Bedrock class

The Llama2Bedrock class is used to load a llama 2 model on Amazon Bedrock and acts as the interface to invoke the llm on Amazon Bedrock. This class can be abstracted to act as a base class for language model interfaces.

For example, this class can be changed to interface with another llm available on Amazon Bedrock (Claude v2, Command, etc.).

`class Llama2Bedrock(pb.models.LMMBaseModel):
    """
    Language model class for interfacing with Llama2 models on Amazon Bedrock.

    Inherits from LMMBaseModel and sets up a model interface for Llama2 models on Amazon Bedrock.

    Parameters:
    -----------
    modelId : str
        The Id of the Llama2 model.
    max_gen_len : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0.2).
    top_p : str, optional
        If set to float less than 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. (default is 0.9).
    """
    def __init__(self, modelId, bedrock_runtime, system_prompt=None, max_gen_len=1024, temperature=0.2, top_p=0.9):
        super(Llama2Bedrock, self).__init__(modelId, max_gen_len, temperature, top_p)
        self.modelId = modelId
        self.bedrock_runtime = bedrock_runtime
        self.max_gen_len = max_gen_len
        self.top_p = top_p
        if system_prompt is None:
            self.system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        else:
            self.system_prompt = system_prompt
    
    def predict(self, input_text, **kwargs):
        try:
            input_text = f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>\n{input_text}[/INST]"
            body = json.dumps({"prompt": input_text, "max_gen_len": self.max_gen_len,
                               "temperature": self.temperature, "top_p": self.top_p})
            accept = "application/json"
            contentType = "application/json"

            response = self.bedrock_runtime.invoke_model(
                    body=body, modelId=self.modelId, accept=accept, contentType=contentType
            )
            response_body = json.loads(response.get("body").read())
            #print(f"prompt_token_count: {response_body.get('prompt_token_count')}")
            #print(f"generation_token_count: {response_body.get('generation_token_count')}")
            #print(response_body.get("generation"))
            
            return response_body.get("generation")

        except botocore.exceptions.ClientError as error:

            if error.response['Error']['Code'] == 'AccessDeniedException':
                   print(f"\x1b[41m{error.response['Error']['Message']}\
                        \nTo troubeshoot this issue please refer to the following resources.\
                         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")

            else:
                raise error`


### Amazon SageMaker JumpStart

The advanced prompt engineering labs deploy various llama 2 models to create model endpoints on Amazon SageMaker. However, other models available on Amazon SageMaker JumpStart can be used to deploy an endpoint. 

This can be changed by changing the model_id and model_version.

```
model_id, model_version = "meta-textgeneration-llama-2-7b-f", "2.*"
from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(model_id=model_id, model_version=model_version)
predictor = model.deploy()
```


## Quick Start

To begin, just start by cloning the repo in an Amazon SageMaker jupyterlab notebook to deploy your llm and start testing various prompt engineering techniques.

In the demo notebook 'llama-2-chat-prompt-engineering.ipynb', we demonstrate how to use the SageMaker Python SDK to deploy a JumpStart model for Text Generation using the Llama 2 fine-tuned model optimized for dialogue use cases.


## Resources
- https://github.com/facebookresearch/PurpleLlama/tree/main
- https://promptbench.readthedocs.io/en/latest/index.html
