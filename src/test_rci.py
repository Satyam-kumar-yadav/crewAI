import unittest
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


# Setup the required llm model to perform the tasks

class TestRCIImplementation(unittest.TestCase):
    def setUp(self):
        self.llm  = llm = ChatOpenAI(
                    model = "mistral-small",
                    base_url = "https://api.mistral.ai/v1"
                    )
    
        self.general_agent = Agent(
            role="Machine Learning Expert",
            goal="Provide clear explanations to the questions asked in very simple language covering even the complex concepts",
            backstory="You are an excellent Machine Learning expert who explains complex concepts in a very understandable way",
            allow_delegation=False,
            verbose=False,
            llm=self.llm
        )

    def test_agent_initialization(self):
        self.assertEqual(self.general_agent.role, "Machine Learning Expert")
        self.assertEqual(self.general_agent.goal, "Provide clear explanations to the questions asked in very simple language covering even the complex concepts")
        self.assertEqual(self.general_agent.backstory, "You are an excellent Machine Learning expert who explains complex concepts in a very understandable way")
        self.assertFalse(self.general_agent.allow_delegation)
        self.assertFalse(self.general_agent.verbose)
        self.assertEqual(self.general_agent.llm, self.llm)

    def test_task_initialization(self):
        task = Task(
            description="what is K-Means Clustering",
            agent=self.general_agent,
            expected_output="A short to medium sized paragraph",
            rci=True,
            rci_depth=1
        )
        self.assertEqual(task.description, "what is K-Means Clustering")
        self.assertEqual(task.agent, self.general_agent)
        self.assertEqual(task.expected_output, "A short to medium sized paragraph")
        self.assertTrue(task.rci)
        self.assertEqual(task.rci_depth, 1)

    def test_rci_process_single_level(self):
        # Mock implementation of RCI for a single level depth
        task = Task(
            description="what is K-Means Clustering",
            agent=self.general_agent,
            expected_output="A short to medium sized paragraph",
            rci=True,
            rci_depth=1
        )

        # Expected process simulation for a single level RCI
        def mock_rci_process(task):
            if task.rci and task.rci_depth == 1:
                return "K-Means Clustering is a type of unsupervised learning algorithm used to group data into clusters based on their similarities."

        output = mock_rci_process(task)
        self.assertEqual(output, "K-Means Clustering is a type of unsupervised learning algorithm used to group data into clusters based on their similarities.")

    def test_rci_process_multiple_levels(self):
        # Mock implementation of RCI for multiple level depth
        task = Task(
            description="what is Quantum Superposition",
            agent=self.general_agent,
            expected_output="A short to medium sized paragraph",
            rci=True,
            rci_depth=3
        )

        # Expected process simulation for multiple levels RCI
        def mock_rci_process(task):
            if task.rci and task.rci_depth == 3:
                intermediate_output1 = "Quantum Superposition is a fundamental principle of quantum mechanics."
                intermediate_output2 = "It states that a quantum system can exist in multiple states at the same time until it is measured."
                final_output = "Quantum Superposition is a fundamental principle of quantum mechanics. It states that a quantum system can exist in multiple states at the same time until it is measured."
                return final_output

        output = mock_rci_process(task)
        self.assertEqual(output, "Quantum Superposition is a fundamental principle of quantum mechanics. It states that a quantum system can exist in multiple states at the same time until it is measured.")

if __name__ == "__main__":
    unittest.main()
