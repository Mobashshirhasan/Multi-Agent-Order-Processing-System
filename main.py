"""
Multi-Agent Order Processing System

This system uses LangGraph to create a network of AI agents that work together to process customer orders :
- Orchestrator Agent: Manages conversation flow and delegates tasks
- Database Agent: Fetches product information from PostgreSQL
- Inventory Agent: Updates product inventory after orders 
- Billing Agent: Generates PDF bills for customer orders
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage as AssistantMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI 
import psycopg2
from psycopg2 import sql
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Database Configuration
DB_CONFIG = {
    "dbname": " ",
    "user": "postgres",
    "password": " ",
    "host": " ",
    "port": "5432"
}

# Type Definitions
class CustomerInfo(TypedDict):
    name: str
    contact: str

class OrderItem(TypedDict):
    item_id: int
    name: str
    quantity: int
    price: float

class OrderState(MessagesState):
    customer_info: Optional[CustomerInfo]
    selected_items: List[OrderItem]
    order_complete: bool
    total_amount: float

# Initialize the LLM instance once
LLM_INSTANCE = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

def get_llm():
    """Return the pre-initialized OpenAI LLM instance"""
    return LLM_INSTANCE

# Database Tools
def connect_to_db():
    """Establish connection to PostgreSQL database"""
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def fetch_products():
    """Fetch all available products from database"""
    connection = connect_to_db()
    if not connection:
        return "Failed to connect to database"
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, name, price, stock FROM products WHERE stock > 0")
        products = cursor.fetchall()
        
        formatted_products = []
        for prod in products:
            formatted_products.append({
                "id": prod[0],
                "name": prod[1],
                "price": prod[2],
                "stock": prod[3]
            })
        
        cursor.close()
        connection.close()
        return formatted_products
    except Exception as e:
        connection.close()
        return f"Error fetching products: {e}"

def update_inventory(item_id, quantity):
    """Update inventory after order is placed"""
    connection = connect_to_db()
    if not connection:
        return "Failed to connect to database"
    
    try:
        cursor = connection.cursor()
        # First check if we have enough stock
        cursor.execute("SELECT stock FROM products WHERE id = %s", (item_id,))
        current_stock = cursor.fetchone()[0]
        
        if current_stock < quantity:
            cursor.close()
            connection.close()
            return f"Insufficient stock for item {item_id}. Available: {current_stock}, Requested: {quantity}"
        
        # Update the stock
        cursor.execute(
            "UPDATE products SET stock = stock - %s WHERE id = %s",
            (quantity, item_id)
        )
        connection.commit()
        cursor.close()
        connection.close()
        return f"Successfully updated inventory for item {item_id}"
    except Exception as e:
        connection.close()
        return f"Error updating inventory: {e}"

def get_product_details(item_id):
    """Get detailed information about a specific product"""
    connection = connect_to_db()
    if not connection:
        return None
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, name, price, stock FROM products WHERE id = %s", (item_id,))
        product = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if product:
            return {
                "id": product[0],
                "name": product[1],
                "price": product[2],
                "stock": product[3]
            }
        return None
    except Exception as e:
        connection.close()
        print(f"Error getting product details: {e}")
        return None

# PDF Generation Tool
def generate_bill_pdf(customer_info, order_items, total_amount):
    """Generate a PDF bill for the customer order"""
    filename = f"bill_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    
    # Content for the PDF
    elements = []
    styles = getSampleStyleSheet()
    
    # Header
    elements.append(Paragraph("INVOICE", styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Customer Info
    elements.append(Paragraph(f"Customer: {customer_info['name']}", styles['Normal']))
    elements.append(Paragraph(f"Contact: {customer_info['contact']}", styles['Normal']))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Order Items Table
    data = [["Item ID", "Item Name", "Quantity", "Unit Price", "Amount"]]
    
    for item in order_items:
        amount = item["quantity"] * item["price"]
        data.append([
            str(item["item_id"]),
            item["name"],
            str(item["quantity"]),
            f"${item['price']:.2f}",
            f"${amount:.2f}"
        ])
    
    # Add total row
    data.append(["", "", "", "Total:", f"${total_amount:.2f}"])
    
    table = Table(data, colWidths=[60, 180, 70, 100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, -1), (-1, -1), colors.beige),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -2), 1, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ('ALIGN', (3, 1), (-1, -1), 'RIGHT'),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 30))
    
    # Footer
    elements.append(Paragraph("Thank you for your purchase! Please visit again.", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    return filename

# Agent Tools
agent_tools = {
    "fetch_products": {
        "name": "fetch_products",
        "description": "Fetches all available products from the database",
        "function": fetch_products
    },
    "update_inventory": {
        "name": "update_inventory",
        "description": "Updates inventory after an order is placed",
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {"type": "integer", "description": "ID of the product"},
                "quantity": {"type": "integer", "description": "Quantity to reduce from inventory"}
            },
            "required": ["item_id", "quantity"]
        },
        "function": update_inventory
    },
    "generate_bill_pdf": {
        "name": "generate_bill_pdf",
        "description": "Generates a PDF bill for the customer",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "contact": {"type": "string"}
                    }
                },
                "order_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "integer"},
                            "name": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "price": {"type": "number"}
                        }
                    }
                },
                "total_amount": {"type": "number"}
            },
            "required": ["customer_info", "order_items", "total_amount"]
        },
        "function": generate_bill_pdf
    },
    "get_product_details": {
        "name": "get_product_details",
        "description": "Gets detailed information about a specific product",
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {"type": "integer", "description": "ID of the product"}
            },
            "required": ["item_id"]
        },
        "function": get_product_details
    }
}

# Agent Definitions
def create_initial_state() -> OrderState:
    """Create initial state for the conversation"""
    return OrderState(
        messages=[],
        customer_info=None,
        selected_items=[],
        order_complete=False,
        total_amount=0.0
    )

def orchestrator_agent(state: OrderState) -> OrderState:
    """Orchestrator agent that manages the overall conversation flow"""
    llm = get_llm()
    messages = state["messages"]
    
    # System prompt for the orchestrator
    system_prompt = """
    You are the main orchestrator AI assistant for an online store. Your job is to:
    1. Greet customers and collect their name and contact information if not already provided
    2. Help customers browse and select products
    3. Coordinate with specialized agents to process orders
    4. Present information clearly and politely to customers
    
    IMPORTANT: Maintain a friendly, professional tone throughout the conversation.
    """
    
    # Check if we need to collect customer information
    if state["customer_info"] is None:
        # If the last message is from the human and doesn't seem to be providing name/contact
        if messages and isinstance(messages[-1], HumanMessage):
            last_msg = messages[-1].content
            
            # Simple check if this might be initial greeting or not answering our question
            if any(word in last_msg.lower() for word in ["hello", "hi", "hey"]) or \
               not any(word in last_msg.lower() for word in ["name", "contact"]):
                response = llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Create a greeting for a new customer and ask for their name and contact number."}
                ])
                return OrderState(
                    messages=messages + [AssistantMessage(content=response.content)],
                    customer_info=state["customer_info"],
                    selected_items=state["selected_items"],
                    order_complete=state["order_complete"],
                    total_amount=state["total_amount"]
                )
            
            # Try to extract customer info
            try:
                extraction_prompt = f"""
                Extract the customer name and contact number from this message:
                "{last_msg}"
                
                If you can identify the information, return it in this format:
                {{
                  "name": "customer name",
                  "contact": "contact number"
                }}
                
                If you cannot identify both pieces of information, return NULL.
                """
                
                extraction = llm.invoke([
                    {"role": "system", "content": "You are a helpful AI assistant that extracts specific information from text."},
                    {"role": "user", "content": extraction_prompt}
                ])
                
                extracted_text = extraction.content.strip()
                if extracted_text.lower() != "null":
                    # Clean up the response to get valid JSON
                    if "```json" in extracted_text:
                        extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in extracted_text:
                        extracted_text = extracted_text.split("```")[1].strip()
                    
                    customer_info = json.loads(extracted_text)
                    if customer_info.get("name") and customer_info.get("contact"):
                        # Successfully extracted customer info
                        new_state = OrderState(
                            messages=messages + [AssistantMessage(content=f"Thank you {customer_info['name']}! How may I help you today? Would you like to order something?")],
                            customer_info=customer_info,
                            selected_items=state["selected_items"],
                            order_complete=state["order_complete"],
                            total_amount=state["total_amount"]
                        )
                        return new_state
            except Exception as e:
                print(f"Error extracting customer info: {e}")
            
            # If extraction failed, explicitly ask for the information
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Create a greeting for a new customer and ask for their name and contact number."}
            ])
            return OrderState(
                messages=messages + [AssistantMessage(content=response.content)],
                customer_info=state["customer_info"],
                selected_items=state["selected_items"],
                order_complete=state["order_complete"],
                total_amount=state["total_amount"]
            )
    
    # If customer info is collected but no items selected yet, check if they want to order
    if state["customer_info"] and not state["selected_items"]:
        last_message = messages[-1].content if messages else ""
        
        # Check if the user wants to order something
        if isinstance(messages[-1], HumanMessage) and any(word in last_message.lower() for word in ["yes", "order", "buy", "purchase", "get"]):
            # Delegate to database agent to fetch products
            return OrderState(
                messages=messages + [AssistantMessage(content="Let me fetch our available products for you.")],
                next="database_agent",
                customer_info=state["customer_info"],
                selected_items=state["selected_items"],
                order_complete=state["order_complete"],
                total_amount=state["total_amount"]
            )
        else:
            # Create a general response asking if they want to order
            conversation_history = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                for msg in messages[-5:] if isinstance(msg, (HumanMessage, AssistantMessage))
            ])
            
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": 
                    f"Customer name: {state['customer_info']['name']}\n"
                    f"Recent conversation:\n{conversation_history}\n\n"
                    "Create a response asking if they would like to order something from our store."
                }
            ])
            
            return OrderState(
                messages=messages + [AssistantMessage(content=response.content)],
                customer_info=state["customer_info"],
                selected_items=state["selected_items"],
                order_complete=state["order_complete"],
                total_amount=state["total_amount"]
            )
    
    # If products are displayed and customer is selecting items
    if messages and isinstance(messages[-1], HumanMessage) and state["customer_info"]:
        last_msg = messages[-1].content.lower()
        
        # Check if the user is selecting items
        if any(phrase in last_msg for phrase in ["i want", "i'd like", "i would like", "buy", "purchase", "select"]) and \
           (any(word.isdigit() for word in last_msg.split()) or "item" in last_msg):
            
            # Try to extract item selection
            extraction_prompt = f"""
            The customer wants to select items from our inventory. Extract the item IDs and quantities from this message:
            "{messages[-1].content}"
            
            Return the selection in this format:
            {{
              "selections": [
                {{"item_id": 1, "quantity": 2}},
                {{"item_id": 3, "quantity": 1}}
              ]
            }}
            
            If no clear selection can be determined, return an empty list.
            """
            
            extraction = llm.invoke([
                {"role": "system", "content": "You are a helpful AI assistant that extracts specific information from text."},
                {"role": "user", "content": extraction_prompt}
            ])
            
            try:
                extracted_text = extraction.content.strip()
                if "```json" in extracted_text:
                    extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
                elif "```" in extracted_text:
                    extracted_text = extracted_text.split("```")[1].strip()
                
                selection_data = json.loads(extracted_text)
                
                if selection_data.get("selections"):
                    # Process each selected item
                    selected_items = []
                    total_amount = 0.0
                    
                    for selection in selection_data["selections"]:
                        item_id = selection.get("item_id")
                        quantity = selection.get("quantity", 1)
                        
                        # Get product details
                        product = get_product_details(item_id)
                        if product:
                            selected_items.append({
                                "item_id": item_id,
                                "name": product["name"],
                                "quantity": quantity,
                                "price": product["price"]
                            })
                            total_amount += product["price"] * quantity
                    
                    if selected_items:
                        # Items selected, update inventory
                        confirmation = f"You've selected {len(selected_items)} item(s) for a total of ${total_amount:.2f}. Let me update our inventory."
                        
                        return OrderState(
                            messages=messages + [AssistantMessage(content=confirmation)],
                            next="inventory_agent",
                            customer_info=state["customer_info"],
                            selected_items=selected_items,
                            order_complete=False,
                            total_amount=total_amount
                        )
            except Exception as e:
                print(f"Error processing selection: {e}")
        
        # Check if they want to complete the order
        if state["selected_items"] and not state["order_complete"] and \
           any(phrase in last_msg for phrase in ["confirm", "place order", "buy now", "complete", "checkout"]):
            confirmation = "Thank you for your order! I'll generate your bill now."
            
            return OrderState(
                messages=messages + [AssistantMessage(content=confirmation)],
                next="billing_agent",
                customer_info=state["customer_info"],
                selected_items=state["selected_items"],
                order_complete=True,
                total_amount=state["total_amount"]
            )
    
    # Default behavior: continue conversation
    conversation_history = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
        for msg in messages[-5:] if isinstance(msg, (HumanMessage, AssistantMessage))
    ])
    
    context = {
        "customer_info": state["customer_info"],
        "has_selected_items": len(state["selected_items"]) > 0,
        "total_items": len(state["selected_items"]),
        "total_amount": state["total_amount"]
    }
    
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": 
                f"Context: {json.dumps(context)}\n"
                f"Recent conversation:\n{conversation_history}\n\n"
                "Generate a helpful response that keeps the conversation going and helps the customer complete their purchase."
            }
        ]
    )
    
    return OrderState(
        messages=messages + [AssistantMessage(content=response.content)],
        customer_info=state["customer_info"],
        selected_items=state["selected_items"],
        order_complete=state["order_complete"],
        total_amount=state["total_amount"]
    )

def database_agent(state: OrderState) -> OrderState:
    """Agent responsible for fetching product information from the database"""
    llm = get_llm()
    messages = state["messages"]
    
    # Fetch products from database
    products = fetch_products()
    
    if isinstance(products, str):  # Error message
        return OrderState(
            messages=messages + [AssistantMessage(content=f"I'm sorry, I couldn't fetch our product list. {products}")],
            next="orchestrator_agent",
            customer_info=state["customer_info"],
            selected_items=state["selected_items"],
            order_complete=state["order_complete"],
            total_amount=state["total_amount"]
        )
    
    # Format product information for display
    system_prompt = """
    You are a database assistant responsible for presenting product information clearly.
    Format the information in a readable way and encourage the customer to make a selection.
    """
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": 
            f"Format this product information for the customer {state['customer_info']['name']}:\n{json.dumps(products)}\n\n"
            "Create a nicely formatted list of products with their ID, name, price, and available stock. "
            "Ask the customer which items they'd like to purchase and in what quantities."
        }
    ])
    
    return OrderState(
        messages=messages + [AssistantMessage(content=response.content)],
        next="orchestrator_agent",
        customer_info=state["customer_info"],
        selected_items=state["selected_items"],
        order_complete=state["order_complete"],
        total_amount=state["total_amount"]
    )

def inventory_agent(state: OrderState) -> OrderState:
    """Agent responsible for updating inventory after items are selected"""
    messages = state["messages"]
    selected_items = state["selected_items"]
    
    # Update inventory for each selected item
    inventory_results = []
    for item in selected_items:
        result = update_inventory(item["item_id"], item["quantity"])
        inventory_results.append(result)
    
    # Check if there were any inventory issues
    if any("Insufficient stock" in result for result in inventory_results):
        # There's an inventory problem
        error_items = [result for result in inventory_results if "Insufficient stock" in result]
        error_message = "I'm sorry, but we have some inventory issues:\n" + "\n".join(error_items)
        
        return OrderState(
            messages=messages + [AssistantMessage(content=error_message)],
            next="orchestrator_agent",
            customer_info=state["customer_info"],
            selected_items=[],  # Clear selected items due to inventory issues
            order_complete=False,
            total_amount=0.0
        )
    
    # All items successfully updated in inventory
    success_message = (
        "Your items have been reserved in our system:\n" + 
        "\n".join([f"- {item['quantity']}x {item['name']} (${item['price']:.2f} each)" for item in selected_items]) +
        f"\n\nTotal: ${state['total_amount']:.2f}\n\nWould you like to complete your order?"
    )
    
    return OrderState(
        messages=messages + [AssistantMessage(content=success_message)],
        next="orchestrator_agent",
        customer_info=state["customer_info"],
        selected_items=selected_items,
        order_complete=False,
        total_amount=state["total_amount"]
    )

def billing_agent(state: OrderState) -> OrderState:
    """Agent responsible for generating bills"""
    llm = get_llm()
    messages = state["messages"]
    
    # Generate PDF bill
    try:
        bill_file = generate_bill_pdf(
            state["customer_info"],
            state["selected_items"],
            state["total_amount"]
        )
        
        # Use the bill_message variable in the response
        bill_message = (
            f"Your bill has been generated successfully! You can find it here: {bill_file}\n\n"
            f"Total amount: ${state['total_amount']:.2f}\n\n"
            "Thank you for your purchase! Please visit again."
        )
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": bill_message}
        ])
        
        system_prompt = """
        You are a friendly billing assistant. Create a warm thank you message for customers who've completed their purchase.
        Include information about their bill and express gratitude for their business.
        """
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": 
                f"Customer name: {state['customer_info']['name']}\n"
                f"Order total: ${state['total_amount']:.2f}\n"
                f"Bill file: {bill_file}\n\n"
                "Create a warm thank you message that mentions the bill was generated, provides the total, "
                "and encourages them to visit again."
            }
        ])
        
        return OrderState(
            messages=messages + [AssistantMessage(content=response.content)],
            next="orchestrator_agent",
            customer_info=state["customer_info"],
            selected_items=state["selected_items"],
            order_complete=True,
            total_amount=state["total_amount"]
        )
    except Exception as e:
        error_message = f"I apologize, but there was an issue generating your bill: {str(e)}"
        return OrderState(
            messages=messages + [AssistantMessage(content=error_message)],
            next="orchestrator_agent",
            customer_info=state["customer_info"],
            selected_items=state["selected_items"],
            order_complete=False,
            total_amount=state["total_amount"]
        )

# Route messages based on state
def router(state: OrderState):
    """Routes the conversation to the appropriate agent"""
    # If there's a next agent specified, route to that
    if "next" in state:
        return state["next"]
    
    # Default to orchestrator
    return "orchestrator_agent"

# Set up the LangGraph
def build_graph():
    """Build the LangGraph for the multi-agent system"""
    workflow = StateGraph(OrderState)
    
    # Add nodes
    workflow.add_node("orchestrator_agent", orchestrator_agent)
    workflow.add_node("database_agent", database_agent)
    workflow.add_node("inventory_agent", inventory_agent)
    workflow.add_node("billing_agent", billing_agent)
    
    # Add edges
    workflow.add_conditional_edges("orchestrator_agent", router)
    workflow.add_conditional_edges("database_agent", router)
    workflow.add_conditional_edges("inventory_agent", router)
    workflow.add_conditional_edges("billing_agent", router)
    
    # Set entry point
    workflow.set_entry_point("orchestrator_agent")
    
    # Compile the graph
    return workflow.compile()

# Main application
def main():
    """Main function to run the multi-agent system"""
    print("Initializing Multi-Agent Order Processing System...")
    
    # Create and compile the graph
    graph = build_graph()
    
    # Initialize state
    state = create_initial_state()
    
    # Welcome message
    print("\n=== Welcome to the AI-Powered Order System ===\n")
    print("System: Hello! I'm your AI shopping assistant.")
    
    # Main conversation loop
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nSystem: Thank you for using our service. Goodbye!")
            break
        
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        
        # Process through the graph
        try:
            state = graph.invoke(state)
            
            # Display assistant's response
            if state["messages"] and isinstance(state["messages"][-1], AssistantMessage):
                print(f"\nAssistant: {state['messages'][-1].content}")
            else:
                print("\nAssistant: I'm processing your request...")
        except Exception as e:
            print(f"\nSystem Error: {str(e)}")
            print("Assistant: I apologize for the technical difficulties. Let's try again.")

if __name__ == "__main__":
    main()
