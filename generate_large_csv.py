import csv
import random
from datetime import datetime, timedelta

def generate_large_csv(filename, num_rows):
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['date', 'stock', 'price', 'volume']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        start_date = datetime(2020, 1, 1)
        for _ in range(num_rows):
            date = start_date + timedelta(days=random.randint(0, 365*3))  # 3 years of data
            stock = random.choice(stocks)
            price = round(random.uniform(50, 1000), 2)
            volume = random.randint(1000, 1000000)
            
            writer.writerow({
                'date': date.strftime('%Y-%m-%d'),
                'stock': stock,
                'price': price,
                'volume': volume
            })
    
    print(f"Generated {num_rows} rows of data in {filename}")

if __name__ == "__main__":
    generate_large_csv('large_stock_data.csv', 10000000)  # Generates 10 million rows