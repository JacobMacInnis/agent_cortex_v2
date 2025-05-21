from agent import get_agent


def main():
    """
    Main function to run the agent.
    """
    print("Welcome to Agent Cortex! Type 'exit' or 'quit' to stop.")

    agent = get_agent()


    while True:

        try: 
            query = input("\n⟁ You: ")
            if query.lower() in ["exit", "quit"]:
                print("Shutting down Cortex")
                break

            response = agent.invoke({"input": query})
            
            
            print(f"\n⚇ Cortex: {response['output']}\n")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()