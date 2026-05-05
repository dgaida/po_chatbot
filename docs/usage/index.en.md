# Usage

The PO-Chatbot offers two main interfaces for different user groups.

## Student Interface (`chatbot_student.py`)

This interface is intended for daily use by students.

*   **Ask a Question**: Enter your question in the text box.
*   **Filter**: Select your faculty and study program to refine the search.
*   **Response & Sources**: The bot generates an answer and lists the used documents with links.

## Admin Interface (`chatbot_admin.py`)

The admin interface is used for quality assurance (Human-in-the-Loop).

1.  **Pending Queue**: Lists all requests that have not yet been validated.
2.  **Detail View**: Admins see the question, the AI response, and the retrieved context.
3.  **Actions**:
    *   **Approve**: The answer is correct.
    *   **Edit & Approve**: Manually correct the answer before approval.
    *   **Reject**: The answer is incorrect or misleading.

## Evaluation Scripts

The `evaluation/` directory contains scripts for automated performance measurement:

*   `evaluate_rag.py`: Runs test questions against different models/configs.
*   `analyze_all_phases.py`: Aggregates the results of the various test phases.
