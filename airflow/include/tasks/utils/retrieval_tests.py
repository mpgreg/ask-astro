import pandas as pd
import os
import json
import uuid

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.retrievers import MultiQueryRetriever
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Weaviate as WeaviateVectorStore
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langsmith import Client as LangsmithClient
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from weaviate.client import Client as WeaviateClient

def weaviate_qna(weaviate_client:WeaviateClient, question:str, class_name:str) -> pd.Series:
    """
    This function uses Weaviate's  
    [QnA](https://weaviate.io/developers/weaviate/modules/reader-generator-modules/qna-openai) 
    to answer questions and returns a pandas series of answers and reference documents used in the answer.
    """

    ask = {
                "question": question,
                "properties": ["content"],
                # "certainty": 0.0
            }

    try:
        results = weaviate_client.query.get(
            class_name=class_name, 
            properties=[
                "docLink", 
                "_additional {answer {hasAnswer property result} }"
                ]
                )\
            .with_ask(ask)\
            .with_limit(3)\
            .with_additional(["certainty", "id", "distance"])\
            .do()['data']['Get'][class_name]
        
        answers = []
        for result in results:
            if result['_additional']['answer']['hasAnswer'] is True:
                answers.append(result['_additional']['answer']['result'])
            else:
                answers.append('')
        answers = '\n-----\n'.join(answers)
        
        references = [result['docLink'] for result in results]
        references = '\n'.join(references)

    except Exception as e:
        print(e)
        answers=[]
        references=[]
    
    return pd.Series([answers, references])

def generate_crc(
        weaviate_client:WeaviateClient, 
        question:str, 
        class_name:str, 
        ts_nodash:str,
        send_feedback:bool = False
    ) -> pd.Series:
    """
    This function uses LangChain's 
    [ConversationalRetrievalChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html)
    along with the 
    [MultiQueryRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html)
    to answer a question with retrieval from the specified corpus (class_name) of documents in weaviate. The
    results are uploaded to LangSmith with a metadata field request_id= "test_baseline".

    The function returns a pandas series of an answer, reference documents used to answer it and a link to 
    the langsmith feedback.
    """

    langsmith_link_template = "https://smith.langchain.com/o/{org}/projects/p/{project}?peek={feedback_id}"

    chat_history = {"question": question,
                    "chat_history": [],
                    "messages": [],
                    }

    with open("include/tasks/utils/combine_docs_chat_prompt.txt") as system_prompt_fd:
        """Load system prompt template from a file and structure it."""
        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt_fd.read()),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]

    docsearch = WeaviateVectorStore(
        client=weaviate_client,
        index_name=class_name,
        text_key='content',
        attributes=['docLink'],
    )

    retriever = MultiQueryRetriever.from_llm(
        llm=AzureChatOpenAI(
            **json.loads(os.environ['AZURE_OPENAI_USEAST_PARAMS']),
            deployment_name="gpt-35-turbo",
            temperature="0.0",
        ),
        retriever=docsearch.as_retriever(),
    )
        
    answer_question_chain = ConversationalRetrievalChain(
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        question_generator=LLMChain(
            llm=AzureChatOpenAI(
                **json.loads(os.environ['AZURE_OPENAI_USEAST_PARAMS']),
                deployment_name="gpt-35-turbo-16k",
                temperature="0.3",
            ),
            prompt=CONDENSE_QUESTION_PROMPT,
        ),
        combine_docs_chain=load_qa_chain(
            llm=AzureChatOpenAI(
                **json.loads(os.environ['AZURE_OPENAI_USEAST2_PARAMS']),
                deployment_name="gpt-4-32k",
                temperature="0.5",
            ),
            chain_type="stuff",
            prompt=ChatPromptTemplate.from_messages(messages),
        ),
    )

    try:
        results = answer_question_chain(inputs=chat_history, 
                                        metadata={"request_id": "test_baseline"})

        if send_feedback is True: 
            feedback = LangsmithClient().create_feedback(
                key="correctness",
                run_id=str(uuid.uuid4()),
                score=0,
                source_info={"dag_run": ts_nodash},
            )
            langsmith_link = langsmith_link_template.format(
                org=os.environ.get('LANGCHAIN_ORG', ''),
                project=os.environ.get('LANGCHAIN_PROJECT_ID', ''),
                feedback_id=feedback.id)
        else: 
            langsmith_link=''

        answers = results['answer']
        references = [result.metadata['docLink'] for result in results['source_documents']]
        references = '\n'.join(list(dict.fromkeys(references)))

    except Exception as e:
        print(e)
        answers=[]
        references=[]
        langsmith_link=''
    
    return pd.Series([answers, references, langsmith_link])

def generate_hybrid_crc(
        weaviate_client:WeaviateClient, 
        question:str, 
        class_name:str, 
        ts_nodash:str,
        send_feedback:bool = False
    ) -> pd.Series:
    """
    This function uses LangChain's 
    [ConversationalRetrievalChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html)
    along with the 
    [MultiQueryRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html)
    to answer a question with retrieval from the specified corpus (class_name) of documents in weaviate. The 
    weaviate search uses a hybrid search approach. The
    results are uploaded to LangSmith with a metadata field request_id= "test_baseline".

    The function returns a pandas series of an answer, reference documents used to answer it and a link to 
    the langsmith feedback.
    """

    langsmith_link_template = "https://smith.langchain.com/o/{org}/projects/p/{project}?peek={feedback_id}"

    chat_history = {"question": question,
                    "chat_history": [],
                    "messages": [],
                    }

    with open("include/tasks/utils/combine_docs_chat_prompt.txt") as system_prompt_fd:
        """Load system prompt template from a file and structure it."""
        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt_fd.read()),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]

    docsearch = WeaviateHybridSearchRetriever(
        client=weaviate_client,
        index_name=class_name,
        text_key='content',
        attributes=['docLink'],
        create_schema_if_missing=True,
    )

    retriever = MultiQueryRetriever.from_llm(
        llm=AzureChatOpenAI(
            **json.loads(os.environ['AZURE_OPENAI_USEAST_PARAMS']),
            deployment_name="gpt-35-turbo",
            temperature="0.0",
        ),
        retriever=docsearch,
    )
        
    answer_question_chain = ConversationalRetrievalChain(
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        question_generator=LLMChain(
            llm=AzureChatOpenAI(
                **json.loads(os.environ['AZURE_OPENAI_USEAST_PARAMS']),
                deployment_name="gpt-35-turbo-16k",
                temperature="0.3",
            ),
            prompt=CONDENSE_QUESTION_PROMPT,
        ),
        combine_docs_chain=load_qa_chain(
            llm=AzureChatOpenAI(
                **json.loads(os.environ['AZURE_OPENAI_USEAST2_PARAMS']),
                deployment_name="gpt-4-32k",
                temperature="0.5",
            ),
            chain_type="stuff",
            prompt=ChatPromptTemplate.from_messages(messages),
        ),
    )

    try:
        results = answer_question_chain(inputs=chat_history, 
                                        metadata={"request_id": "test_baseline"})

        if send_feedback is True: 
            feedback = LangsmithClient().create_feedback(
                key="correctness",
                run_id=str(uuid.uuid4()),
                score=0,
                source_info={"dag_run": ts_nodash},
            )
            langsmith_link = langsmith_link_template.format(
                org=os.environ.get('LANGCHAIN_ORG', ''),
                project=os.environ.get('LANGCHAIN_PROJECT_ID', ''),
                feedback_id=feedback.id)
        else: 
            langsmith_link=''

        answers = results['answer']
        references = [result.metadata['docLink'] for result in results['source_documents']]
        references = '\n'.join(list(dict.fromkeys(references)))

    except Exception as e:
        print(e)
        answers=[]
        references=[]
        langsmith_link=''
    
    return pd.Series([answers, references, langsmith_link])