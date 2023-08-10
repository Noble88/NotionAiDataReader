#TODO | [Research] find how to make ai use more than the (suspected) 4 documents it pulls from.as
#   More info: "Vectorstores and Embedding" section in https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/4/vectorstores-and-embedding
#     Look into: [vectordb.similarity_search] & [doucument pulling]

#TODO | [Research] find out how to change metadata of sources [MB]
#   Current Info: metadata can be used for filterting purpose
#   More Info: retrieval, , https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/5/retrieval
#   Note: using "notionDB[0].metadata" has shows metadata of page
#   Details: can make filter terms in meta data and can once done should take out the tag catagories to free up
#     tokens allowing to go through and use more documents

#TODO | [IDEA] If Metadata Modification doesn't work, looking into "Context aware splitting"
#   More Info: Near end of lesson in "Document Splitting" Section on -> https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/3/document-splitting

import os
import sys

import openai

sys.path.append('../..')

#region Ai and Debugging Modification and Settings
  #region Debugger Method
def debug(section):
  debugSections = {
    "loadDocuments": True,
    "splittingDocuments": True,
    "embeddings": True,
    "embeddingTests": False, #under development & not ready to be used
    "printResults": False,
    "showSourceDocuments": False,
    #region Testing Grounds
    "testingGrounds": False,
    "metadataTest": True,
    #endregion
  }
  if section in debugSections: #Valid Key
    return debugSections[section]
  else: #Invalid Key
    errorPrintVariable(
      "MINOR",
      "section",
      section,
      "bool: False",
      "[Ai and Debugging Modification and Settings] / [Debugger Method]",
      True,
      "debug(section)",
      "Could not find matching key using parameter."
    )
    return debugSections.get(section, False) #I think can just say -> return False
# demo print line length: "\n\t-------------------------"
#endregion
  #region Error Method
    #region Error Category Explanations
      #region error severity
#FATAL   : Will/Typically break the code. Need immediate fix
#CRITICAL: Can lead to incorrect behavior or critical issues. Should fix before testing.
#MAJOR   : Have a large impact on program behavior, leads to incorrect results. Might hinder testing and warrant a fix
#MINOR   : Doesn't significantly affect functionality, but can aid in bad results.
#           OR Error won't accomplish wanted task but doesn't have an effect on the overall function of the code.
#WARNING : Can be a potential issue
      #endregion
    #endregion

def errorPrintGeneric(errorSeverity, regionLocation, details):
  print("\n---------- Error [",errorSeverity,"] - Error Type [Generic] ----------",
        "\n\tRegion Location: ", regionLocation,
        "\n\tDetails:         ", details
        )
  print("---------------------------------------------------------------")
def errorPrintVariable(errorSeverity, variableName, variableContents, defaultReturn, regionLocation, isInMethod, MethodName, details):
  print("\n---------- Error [",errorSeverity,"] - Error Type [Bad Variable] ----------",
        "\n\tVariable:        ", variableName,
        "\n\tVariable Data:   ", variableContents,
        "\n\tReturning:       ", defaultReturn,
        "\n\tRegion Location: ", regionLocation,
        "\n\tIn Method:       ", isInMethod
        )
  if isInMethod:
    print("\tMethod Name:     ", MethodName)
  print("\tDetails & Info:  ", details)
  print("---------------------------------------------------------------")
#length ruler -> "\n\t------------------: "
#endregion
#TODO | [Organize] reorganize for better visual aid for regions [Ai Input Customization] & [AI Input & Modifiers]
  #region Ai Input Customization
def aiInputSettings(modifier):
  definedParamters = {
    "splittingType": "token", #Available Types: token, character
    "remakeEmbeddingData": True,
    "saveConversation": True, #Changing to false breaks stuff...
  }
  if modifier in definedParamters: #Valid Key
    return definedParamters[modifier]
  else: #Invalid Key
    errorPrintVariable(
      "FATAL",
      "modifier",
      modifier,
      "NONE",
      "[Ai and Debugging Modification and Settings] / [Ai Input Customization]",
      True,
      "aiInputSettings(modifier)",
      "Could not find matching key using parameter."
    )
  return definedParamters.get(modifier, None)
#endregion
  #region AI Input & Modifiers
documentPulling = 5 # Note: varaible is Nonfunctional
temp = 0.05

#TODO | [Research] Learned Prompt Engneering to cut down token amount in template (find what are filler words and how to say what is only necessary)
template = """Review the sections instructions & information when answering prompts...
{information}

{context}

{chat_history}
Question: {question}
Helpful Answer:"""
  #endregion
#endregion

#region Open AI Info
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']
#endregion

#region Load Documents - Notion Database
  #region Load Documents
from langchain.document_loaders import NotionDirectoryLoader
NotionData = NotionDirectoryLoader("NotionData")
notionDB = NotionData.load()
  #endregion
  #region Debugger
if debug("loadDocuments"):
  print("\n----- Debugger: Load Documents - Notion Database -----",
        "\n\tTotal Pages:             ", len(notionDB)
        )
  print("---------------------------------------------------------------")
#endregion
#endregion

#region Splitting Documents
  #region Imports
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
  #endregion
  #region Splitter Method
def dataSplitter(): #Available Types: "token", "character"
  splitType = aiInputSettings("splittingType")
  if splitType == "token":
    text_splitter = TokenTextSplitter(
      chunk_size=1750,
      chunk_overlap=30
    )
  elif splitType == "character":
    text_splitter = CharacterTextSplitter(
      separator=" ",
      chunk_size=1000,
      chunk_overlap=50,
      length_function=len
    )
  else:
    errorPrintVariable("FATAL",
                       "splittingType",
                       splitType,
                       "NONE"
                       "[Ai and Debugging Modification and Settings] / [Ai Input Customization]",
                       True,
                       "aiInputSettings(modifier)",
                       "Variable doesn't match any comparison variables"
                       )
  return text_splitter.split_documents(notionDB)
#endregion
splitNotionData = dataSplitter()
  #region Debugging
if debug("splittingDocuments"):
  print("\n----- Debugger: Splitting Documents - Active Type [", aiInputSettings("splittingType").capitalize(), "] Splitting -----",
        "\n\tToken Split:             ", len(splitNotionData),
        )
  print("---------------------------------------------------------------")
#endregion
#endregion

#region Embeddings
  #region Imports & Needed Initialization
from langchain.embeddings.openai import OpenAIEmbeddings #Note: Used inside of createEmbeddings method
from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/' #This is where the data of any previously saved embeddings go (I think...)
#endregion
  # region Remove Chroma File Method
    #(Method used to Reset "persist_directory" [I think...] )
import shutil
def removeChroma():
  chromaDebug="N/A"
  try:
    shutil.rmtree("./docs/chroma")
    chromaDebug = "Removed Directory Successfully."
  except FileNotFoundError:
    chromaDebug = "Directory Doesn't Exist!"
  except Exception as e:
    errorPrintGeneric(
      "CRITICAL",
      "[Embeddings] / [Remove Chroma File Method]"
      f"An error occurred while removing 'chroma' directory: {e}"
    )
  return chromaDebug
if aiInputSettings("remakeEmbeddingData"):
  chromaDebug = removeChroma()
else:
  chromaDebug = "Did not need deletion"
# endregion
  #region Create Embeddings
def createEmbeddings():
  embedding = OpenAIEmbeddings()
  if aiInputSettings("remakeEmbeddingData"):
    removeChroma()
    generatedEmbed = Chroma.from_documents(
      documents=splitNotionData,
      embedding=embedding,
      persist_directory=persist_directory
    )
  else:
    generatedEmbed = Chroma(
      persist_directory=persist_directory,
      embedding_function=embedding
    )
  return generatedEmbed
vectordb = createEmbeddings()
vectordb.persist()
#endregion
  # region Debugging
if debug("embeddings"):
  print("\n----- Debugger: Embeddings - Remade Embeddings [", aiInputSettings("remakeEmbeddingData"), "] -----",
        "\n\tEmbedding chunk amount:  ", vectordb._collection.count(),
        "\n\t\tNote: [Embedding chunk amount] should equal [Token Split]",
        "\n\tChroma Status:           ", chromaDebug,
        )
  print("---------------------------------------------------------------")

#TODO | [ADD] Make a debug that test search features of VecotrDB and Embedding
if debug("embeddingTests"):
  question = "[INSERT YOUR QUESTION HERE]"
  print("\nDebugger: Embeddings Tests - Pulling [",documentPulling,"] Documents",
        "\n\tQuestion:                ", question,
        "\n\tResults:                 ", vectordb.similarity_search(question, k=documentPulling),
        )
#endregion
#endregion

#region Output
  #region Construct Language Model
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
  model_name="gpt-3.5-turbo",
  temperature=temp
)
#endregion
  #region Build prompt
from langchain.prompts import PromptTemplate
QA_CHAIN_PROMPT = PromptTemplate.from_template(template) #Note: template is in region path [Ai and Debugging Modification and Settings] / [AI Input & Modifiers]
#endregion
  #region QA Chain (AI Response)
def createQaChain():
  if aiInputSettings("saveConversation"):
    #region Create Memory
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(
      memory_key='chat_history',
      return_messages=True
    )
    #endregion
    #region Create QA Retrieval (w/Memory)
    from langchain.chains import ConversationalRetrievalChain
    qa = ConversationalRetrievalChain.from_llm(
      llm,
      retriever=vectordb.as_retriever(),
      #return_source_documents=debug("showSourceDocuments"),
      memory=memory
    )
    #endregion
  else:
    #region Create QA Retrieval (w/o Memory)
    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(
      llm,
      retriever=vectordb.as_retriever(),
      #return_source_documents=debug("showSourceDocuments"),
      chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    #endregion
  return qa
#endregion
qa=createQaChain()

#endregion

#region Testing Grounds
if debug("testingGrounds"):
  print("\n----- Debugger: Testing Grounds - Active Tests:",
        "[ Metadata:", debug("metadataTest"), "] -----")
  if debug("metadataTest"):
    print("\tMetadata Test:",
          "\n\t\tFirst Page Meta Data:    ", notionDB[0].metadata,
          )
  print("---------------------------------------------------------------")
#endregion

#region Terminal Chat System
def get_ai_response(question):
  result = qa({"question": question})
  if debug("printResults"):
    print("\nDebugger: Terminal Chat System: ",
          "\n\tPrinted Results:         ", result,
          )
  return result['answer']

#region Loop
print("Welcome to the AI chat system!")
while True:
  user_input = input("You: ")

  #region Exit Chat
  if user_input.lower() in ["quit", "exit", "bye"]:
      print("Goodbye!")
      break
  #endregion

  # region AI Response
  ai_response = get_ai_response(user_input)
  print("AI: ", ai_response)
  #endregion
#endregion
#endregion