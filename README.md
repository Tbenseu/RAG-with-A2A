# Steps on gRPC
1. Create proto file
2. Generate protobuf file
    - Sync
        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. rag_agent.proto
        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. web_search_agent.proto
        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. orchestrator_agent.proto
    - Async
        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. rag_agent/rag_agent.proto
        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. web_search_agent/web_search_agent.proto
        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. orchestrator/orchestrator_agent.proto
3. Develop your gRPC script for each Agent and integrate your tools
4. Develop the streamlit frontend app
5. Run the Chroma db Script
    ```chroma run --path ./data --port 5000```
5. Run the MultiAgent 
    run ```Python main.py``` in individual agent folder
6. Run yout Streamlit Script
    ```streamlit run app.py```
