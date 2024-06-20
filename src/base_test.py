from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Setup the required llm model to perform the tasks

llm = ChatOpenAI(
    model = "mistral-small",
    base_url = "https://api.mistral.ai/v1"
    )

### Machine Learning Expert 

# general_agent = Agent(
#     role="Machine Learning Expert",
#     goal="""Provide clear explanations to the questions asked in a very simple language covering even the complex concepts""",
#     backstory="""You are an excellent Machiene Learning expert who explains complex concepts in a very understandable way""",
#     allow_delegation=False,
#     verbose=False,
#     llm=llm,
# )

# task = Task(
#     description="""what is K-Means Clustering""",
#     agent=general_agent,
#     expected_output="A short to meduim sized paragraph",
#     rci = True,
#     rci_depth = 1
# )

### Healthcare Expert
# general_agent = Agent(
#     role="Healthcare Expert",
#     goal="""Provide clear explanations to the questions asked in very simple language covering even the complex concepts""",
#     backstory="""You are an excellent Healthcare expert who explains complex concepts in a very understandable way""",
#     allow_delegation=False,
#     verbose=False,
#     llm=llm,
# )

# task = Task(
#     description="""what is hypertension""",
#     agent=general_agent,
#     expected_output="A short to medium sized paragraph",
#     rci=True,
#     rci_depth=1
# )

### Quantum Computing Expert
general_agent = Agent(
    role="Quantum Computing Expert",
    goal="""Provide clear explanations to the questions asked in very simple language covering even the complex concepts""",
    backstory="""You are an excellent Quantum Computing expert who explains complex concepts in a very understandable way""",
    allow_delegation=False,
    verbose=False,
    llm=llm,
)

task = Task(
    description="""Explain the principle of quantum superposition and how it is utilized in quantum computing""",
    agent=general_agent,
    expected_output="A short to medium sized paragraph",
    rci=True
    )

mathematics_agent = Agent(
    role="Mathematics Expert",
    goal="Provide clear explanations to mathematical questions",
    backstory="You are an expert in mathematics who explains concepts clearly and concisely",
    allow_delegation=False,
    verbose=False,
    llm=llm,
)

# Example mathematics task
math_task = Task(
    description="Explain the concept of integration by parts",
    agent=mathematics_agent,
    expected_output="A short to medium sized paragraph",
    rci=True,
    rci_depth=1
)

# Define your physics agent
physics_agent = Agent(
    role="Physics Expert",
    goal="Provide clear explanations to physics questions",
    backstory="You are an expert in physics who explains concepts clearly and concisely",
    allow_delegation=False,
    verbose=False,
    llm=llm,
)

# Example physics task
physics_task = Task(
    description="Describe the theory of relativity",
    agent=physics_agent,
    expected_output="A short to medium sized paragraph",
    rci=True,
    rci_depth=1
)

if __name__ == "__main__":
    # Create a Crew instance with all agents and tasks
    # crew1 = Crew(agents=[general_agent, mathematics_agent, physics_agent], tasks=[task, math_task, physics_task], verbose=2)
    # results = crew.kickoff()
    # print(results)


    crew1 = Crew(agents=[general_agent], tasks=[task], verbose=2)
    # crew2 = Crew(agents=[mathematics_agent], tasks=[math_task], verbose=2)
    # crew3 = Crew(agents=[physics_agent], tasks=[physics_task], verbose=2)
    # Kick off task
    results1 = crew1.kickoff()
    # results2 = crew2.kickoff()
    # results3 = crew3.kickoff()


    # Print results
    print(results1)
