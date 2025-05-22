from agent import get_agent

def format_chat_history(messages):
    lines = []
    for msg in messages:
        role = "Human" if msg.type == "human" else "AI"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)

def inject_memory_into_input(history: str, user_input: str) -> str:
    return f"{history}\n\nHuman: {user_input}"


def main():
    """
    Main function to run the agent.
    """
    print("Welcome to Agent Cortex! Type 'exit' or 'quit' to stop.")

    agent, memory = get_agent()


    while True:

        try: 
            query = input("\n⟁ You: ")
            if query.lower() in ["exit", "quit"]:
                print("Shutting down Cortex")
                break
            print(agent.memory.chat_memory.messages) # type: ignore
            formatted_history = format_chat_history(memory.chat_memory.messages)
            full_input = inject_memory_into_input(formatted_history, query)

            response = agent.invoke({
                "input": full_input,
                "chat_history": memory,
            })
            

            print(f"\n⚇ Cortex: {response['output']}\n")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()