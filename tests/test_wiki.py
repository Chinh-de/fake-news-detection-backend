import os
import sys
import asyncio

# Ensure stdout and stderr use UTF-8 to prevent encoding errors on Windows terminal
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass


# Add the Backend directory to Python path so we can import 'app'
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

try:
    from app.services.retrieval_service import retrieval_service
    print("[SUCCESS] Imported retrieval_service successfully.")
except Exception as e:
    print(f"[ERROR] Failed to import retrieval_service: {e}")
    sys.exit(1)

# List of 10 Vietnamese entities to test
entities_to_test = [
    "Tổ chức Y tế Thế giới",
    "WHO",
    "SARSCoV-2",
    "FPT",
    "Vịnh Hạ Long",
    "Sơn Tùng M-TP",
    "Covid-19",
    "Viettel",
    "Hà Nội",
    "Mỹ Tâm"
]

async def run_test():
    print(f"Testing Wikipedia retrieval for {len(entities_to_test)} entities...\n")
    
    # Measure execution time
    start_time = asyncio.get_event_loop().time()
    
    # Run the get_wiki_definitions method from retrieval_service
    results = await retrieval_service.get_wiki_definitions(entities_to_test)
    
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    
    print(f"Completed in {duration:.2f} seconds.")
    print(f"Retrieved information for {len(results)} out of {len(entities_to_test)} entities:\n")
    
    for i, entity in enumerate(entities_to_test, 1):
        if entity in results:
            summary = results[entity]
            # Truncate summary for clean output
            preview = summary[:150] + "..." if len(summary) > 150 else summary
            print(f"{i}. [FOUND] {entity}:")
            print(f"   => {preview}\n")
        else:
            print(f"{i}. [NOT FOUND] {entity}\n")

if __name__ == "__main__":
    asyncio.run(run_test())
