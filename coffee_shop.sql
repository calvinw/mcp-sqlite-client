-- Create the products table
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    description TEXT,
    in_stock INTEGER DEFAULT 1
);

-- Create the orders table
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    customer_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    is_completed INTEGER DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

-- Insert data into products table
INSERT INTO products (product_id, name, category, price, description, in_stock) VALUES
(1, 'Espresso', 'Coffee', 2.50, 'Strong black coffee made by forcing steam through ground coffee beans', 1),
(2, 'Cappuccino', 'Coffee', 3.50, 'Espresso with steamed milk foam', 1),
(3, 'Latte', 'Coffee', 3.75, 'Espresso with steamed milk', 1),
(4, 'Americano', 'Coffee', 2.75, 'Espresso with hot water', 1),
(5, 'Matcha Latte', 'Tea', 4.25, 'Green tea powder with steamed milk', 1),
(6, 'Earl Grey', 'Tea', 2.50, 'Black tea flavored with bergamot', 1),
(7, 'Chocolate Croissant', 'Pastry', 3.25, 'Buttery croissant with chocolate filling', 1),
(8, 'Blueberry Muffin', 'Pastry', 2.95, 'Moist muffin with fresh blueberries', 1),
(9, 'Iced Coffee', 'Coffee', 3.25, 'Chilled coffee served with ice', 1),
(10, 'Chai Latte', 'Tea', 3.95, 'Spiced tea with steamed milk', 1);

-- Insert data into orders table
INSERT INTO orders (order_id, product_id, customer_name, quantity, order_date, is_completed) VALUES
(1, 2, 'John Smith', 1, '2025-04-13 08:15:00', 1),
(2, 7, 'John Smith', 1, '2025-04-13 08:15:00', 1),
(3, 5, 'Emma Johnson', 1, '2025-04-13 09:22:00', 1),
(4, 10, 'Michael Brown', 2, '2025-04-13 10:05:00', 1),
(5, 1, 'Sarah Davis', 1, '2025-04-13 10:30:00', 1),
(6, 3, 'David Wilson', 1, '2025-04-13 11:15:00', 0),
(7, 8, 'David Wilson', 2, '2025-04-13 11:15:00', 0),
(8, 9, 'Jennifer Lee', 1, '2025-04-13 11:42:00', 0),
(9, 4, 'Robert Taylor', 1, '2025-04-13 12:10:00', 0),
(10, 6, 'Lisa Anderson', 3, '2025-04-13 12:25:00', 0);

-- Example query: Get all pending orders with product details
SELECT 
    o.order_id, 
    p.name AS product_name, 
    o.customer_name, 
    o.quantity, 
    p.price, 
    (o.quantity * p.price) AS total_price,
    o.order_date
FROM 
    orders o
JOIN 
    products p ON o.product_id = p.product_id
WHERE 
    o.is_completed = 0
ORDER BY 
    o.order_date;

-- Example query: Get total sales by category
SELECT 
    p.category, 
    SUM(o.quantity) AS items_sold, 
    SUM(o.quantity * p.price) AS total_sales
FROM 
    orders o
JOIN 
    products p ON o.product_id = p.product_id
GROUP BY 
    p.category
ORDER BY 
    total_sales DESC;
