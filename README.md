# Multi-Agent Order Processing System.

A sophisticated e-commerce backend system that uses LangGraph to create a network of AI agents working together to process customer orders efficiently and intelligently.

## 🌟 Features :

- **Orchestrator Agent**: Manages conversation flow and delegates tasks
- **Database Agent**: Fetches product information from PostgreSQL
- **Inventory Agent**: Updates product inventory after orders
- **Billing Agent**: Generates PDF bills for customer orders

## 📋 System Overview :

This project implements a multi-agent architecture using LangGraph to create a seamless order processing system. Each agent has specific responsibilities:

- **Orchestrator Agent**: The main interface that interacts with customers, collects their information, and coordinates with other agents
- **Database Agent**: Handles all database queries to retrieve product information
- **Inventory Agent**: Manages product inventory levels by updating stock after orders are placed
- **Billing Agent**: Generates professional PDF bills for completed orders

## 🔧 Requirements : 

- Python 3.9+
- PostgreSQL database
- OpenAI API key

## 📦 Dependencies :

```
langchain
langgraph
langchain-openai
psycopg2-binary
reportlab
```

## ⚙️ Configuration :

Before running the application, you need to set up your PostgreSQL database and configure the environment:

1. Create a PostgreSQL database with a `products` table that has the following structure:
   - `id` (integer, primary key)
   - `name` (text)
   - `price` (float)
   - `stock` (integer)

2. Update the DB_CONFIG dictionary in `app8.py` with your database credentials:
   ```python
   DB_CONFIG = {
       "dbname": "your_db_name",
       "user": "postgres",
       "password": "your_password",
       "host": "localhost",
       "port": "5432"
   }
   ```

3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```
   or use a `.env` file (add this file to `.gitignore`).

## 🚀 Getting Started :

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/multi-agent-order-system.git
   cd multi-agent-order-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## 💬 Usage :

Once the application is running, you can interact with the system through the terminal:

1. The system will greet you and ask for your name and contact information
2. After providing your details, you can browse available products
3. Select products by specifying item IDs and quantities
4. Confirm your order to generate a PDF bill

Example interaction:

```
=== Welcome to the AI-Powered Order System ===

System: Hello! I'm your AI shopping assistant.
