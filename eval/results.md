# CityWheels Help Desk Agent — Evaluation Results

## Summary

- **Model**: `gemini-3.5-flash`
- **Policy Search Mode**: `auto`
- **Questions Passed**: **14 / 14 (100.0%)**
- **Average Latency per Question**: `10.35s`

## Detailed Question Results

| ID | Category | Question | Tools Called | Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q01** | `policy` | What time do you open on Saturday? | `search_policies` | 4.98s | **PASS** |
| **Q02** | `policy` | How much is an e-bike for a whole day? | `search_policies` | 7.17s | **PASS** |
| **Q03** | `policy` | Can my 14-year-old rent an e-bike? | `search_policies` | 6.51s | **PASS** |
| **Q04** | `policy` | Can I bring my dog in the bike basket? | `search_policies` | 9.68s | **PASS** |
| **Q05** | `policy` | What happens if I return a city bike 2 hours late? | `search_policies`, `search_policies` | 23.53s | **PASS** |
| **Q06** | `refund` | I want to cancel booking CW-1002. How much will I get back? | `get_booking`, `calculate_refund` | 14.36s | **PASS** |
| **Q07** | `refund` | If I cancel booking CW-1003 right now, how much will I get back? | `calculate_refund`, `get_booking` | 20.96s | **PASS** |
| **Q08** | `refund` | I want to cancel booking CW-1004. How much will I get back and how is it refunded? | `get_booking`, `calculate_refund`, `search_policies` | 17.85s | **PASS** |
| **Q09** | `refund` | Can I cancel booking CW-1008 and get a refund? | `get_booking`, `calculate_refund`, `search_policies` | 9.39s | **PASS** |
| **Q10** | `refund` | I want to cancel booking CW-1007. How much will I get back? | `calculate_refund`, `get_booking` | 5.91s | **PASS** |
| **Q11** | `booking` | Can you check the details of booking CW-9999? | `get_booking` | 6.04s | **PASS** |
| **Q12** | `refusal` | What is the capital of Australia? | _(none)_ | 2.77s | **PASS** |
| **Q13** | `refusal` | How do I bake a chocolate cake at home? | _(none)_ | 4.11s | **PASS** |
| **Q14** | `escalation` | There is a storm warning for my rental day. Will I get a full refund automatically? | `search_policies` | 11.68s | **PASS** |

## Retrieval Approach Comparison (BM25 vs Hybrid Embeddings)

| Query | Expected Section | BM25 Top-1 (Time) | Hybrid Top-1 (Time) |
| :--- | :--- | :--- | :--- |
| What time do you open on Saturday? | `When are you open?` | `When are you open?` [OK] (5.09ms) | `When are you open?` [OK] (843.55ms) |
| How much is an e-bike for a whole day? | `Bike rental` | `Bike rental` [OK] (4.43ms) | `Bike rental` [OK] (498.1ms) |
| Can my 14-year-old rent an e-bike? | `Are there age limits?` | `Are there age limits?` [OK] (4.95ms) | `Are there age limits?` [OK] (468.25ms) |
| Can I bring my puppy in the basket? | `Can I take my dog?` | `Can I take my dog?` [OK] (2.7ms) | `Can I take my dog?` [OK] (435.2ms) |
| What if I return a bike 2 hours late? | `Late return` | `Late return` [OK] (3.71ms) | `Late return` [OK] (798.4ms) |
| Storm warning on my rental day | `Bad weather` | `Bad weather` [OK] (3.95ms) | `Bad weather` [OK] (491.28ms) |
| Can I buy a bike from your shop? | `Do you sell bikes?` | `Do you sell bikes?` [OK] (4.04ms) | `Do you sell bikes?` [OK] (472.61ms) |

## Question-by-Question Transcripts

### Q01 (policy) — PASSED
- **Question**: What time do you open on Saturday?
- **Expectation**: Opening hours on Saturdays (seasonal schedule in faq.md)
- **Tools Called**: `['search_policies']`
- **Agent Answer**:
```text
From April to October, CityWheels is open every day, including Saturdays, from **08:00 to 20:00**.

*(For reference, from November to March, opening hours on Saturdays are 09:00 to 17:00).*
```

### Q02 (policy) — PASSED
- **Question**: How much is an e-bike for a whole day?
- **Expectation**: Daily rental price for an electric bike (40 EUR)
- **Tools Called**: `['search_policies']`
- **Agent Answer**:
```text
An electric bike (`e-bike`) costs **40 EUR per day** (up to 24 hours). 

Key details:
- **Daily Rate:** 40 EUR ( rentals longer than 6 hours are automatically charged at the day rate).
- **Deposit:** A refundable deposit of 150 EUR per electric bike is required at booking (returned in full when the bike is returned on time and without damage).
```

### Q03 (policy) — PASSED
- **Question**: Can my 14-year-old rent an e-bike?
- **Expectation**: Age limit check: e-bikes require 16+, city bikes 13+
- **Tools Called**: `['search_policies']`
- **Agent Answer**:
```text
No, a 14-year-old cannot rent an electric bike. 

According to CityWheels policies:
* **Electric bikes** are available for riders from **16 years old** (riders under 18 need a parent or guardian to sign the rental).
* **City bikes** are available from **13 years old** (with a parent or guardian signature for riders under 18).
```

### Q04 (policy) — PASSED
- **Question**: Can I bring my dog in the bike basket?
- **Expectation**: Dogs are not allowed in bike baskets or child seats
- **Tools Called**: `['search_policies']`
- **Agent Answer**:
```text
No, dogs are not allowed in bike baskets or child seats.
```

### Q05 (policy) — PASSED
- **Question**: What happens if I return a city bike 2 hours late?
- **Expectation**: Late return fee is 1.5x hourly price per started hour, taken from deposit
- **Tools Called**: `['search_policies', 'search_policies']`
- **Agent Answer**:
```text
If you return a city bike late, you will be charged **1.5 times the hourly price** for every started hour of delay.

For a city bike, the hourly rate is 4 EUR. Therefore, the late fee is **6 EUR per hour**. If you return the bike 2 hours late, the total fee will be **12 EUR**, which will be deducted from your deposit.
```

### Q06 (refund) — PASSED
- **Question**: I want to cancel booking CW-1002. How much will I get back?
- **Expectation**: Refund for CW-1002 (33h before start): 50% of 80 EUR (40 EUR) + 300 EUR deposit = 340 EUR
- **Tools Called**: `['get_booking', 'calculate_refund']`
- **Agent Answer**:
```text
If you cancel booking CW-1002 now, you will receive a total refund of **340.00 EUR**.

Here is the breakdown of your refund:
*   **Rental price refund:** 40.00 EUR (50% of the 80.00 EUR rental price, as you are cancelling between 24 and 48 hours before the start).
*   **Deposit refund:** 300.00 EUR (100% refund).
*   **Booking fee:** 0.00 EUR (the 2.00 EUR booking fee is non-refundable).

Since you paid by card, the total amount of 340.00 EUR will be refunded to the same card within 5 working days.
```

### Q07 (refund) — PASSED
- **Question**: If I cancel booking CW-1003 right now, how much will I get back?
- **Expectation**: Refund for CW-1003 (6h before start): 0 EUR rental refund + 50 EUR deposit = 50 EUR
- **Tools Called**: `['calculate_refund', 'get_booking']`
- **Agent Answer**:
```text
If you cancel booking CW-1003 right now (2026-06-10 at 09:00), here is the breakdown of your refund:

*   **Rental price refund:** 0.00 EUR (Since the cancellation is less than 24 hours before the rental starts, the rental price is non-refundable).
*   **Deposit refund:** 50.00 EUR.
*   **Booking fee:** The 2.00 EUR booking fee is non-refundable.
*   **Total value returned:** 50.00 EUR.

The total amount of 50.00 EUR will be refunded to your card within 5 working days.
```

### Q08 (refund) — PASSED
- **Question**: I want to cancel booking CW-1004. How much will I get back and how is it refunded?
- **Expectation**: Refund for CW-1004 (paid with voucher, >48h): 23 EUR as new voucher + 50 EUR deposit
- **Tools Called**: `['get_booking', 'calculate_refund', 'search_policies']`
- **Agent Answer**:
```text
If you cancel your booking **CW-1004** now (on June 10, 2026, at 09:00), you will receive a total of **73.00 EUR** back.

Here is the breakdown of your refund:
*   **Rental price refund:** 23.00 EUR (100% refund, as you are cancelling more than 48 hours before the start).
*   **Deposit refund:** 50.00 EUR.
*   **Booking fee:** 0.00 EUR (the 2.00 EUR booking fee is non-refundable).

**How it is refunded:**
Since your booking was paid with a **voucher**:
*   The **rental price (23.00 EUR)** will be refunded as a **new voucher** of the same value. It cannot be refunded as cash or to a card.
*   The **deposit (50.00 EUR)** will be refunded the same way it was paid.

If you would like to proceed with this cancellation or have any other questions, please let me know!
```

### Q09 (refund) — PASSED
- **Question**: Can I cancel booking CW-1008 and get a refund?
- **Expectation**: CW-1008 is in_progress and cannot be cancelled (0 EUR refund)
- **Tools Called**: `['get_booking', 'calculate_refund', 'search_policies']`
- **Agent Answer**:
```text
No, you cannot cancel booking **CW-1008**. 

Booking CW-1008 currently has the status **`in_progress`** (rental period: June 10, 2026, 08:00 – 12:00). According to CityWheels policy, bookings that are `in_progress`, `completed`, or already `cancelled` cannot be cancelled, and no refund is issued.

Here is the breakdown of the refund calculation:
- **Rental Price Refund:** €0.00
- **Deposit Refund:** €0.00
- **Booking Fee:** Non-refundable (€2.00 fee)
- **Total Value Returned:** €0.00
```

### Q10 (refund) — PASSED
- **Question**: I want to cancel booking CW-1007. How much will I get back?
- **Expectation**: Boundary refund for CW-1007 (exactly 48h before start): 100% rental (36 EUR) + 150 EUR deposit = 186 EUR
- **Tools Called**: `['calculate_refund', 'get_booking']`
- **Agent Answer**:
```text
If you cancel booking **CW-1007** now (2026-06-10 09:00), here is the refund breakdown:

* **Rental Price Refund:** **€36.00** (100% refund, returned to your original payment card within 5 working days)
* **Deposit Refund:** **€150.00** (100% refunded)
* **Booking Fee:** The **€2.00** booking fee is **non-refundable**
* **Total Value Returned:** **€186.00**
```

### Q11 (booking) — PASSED
- **Question**: Can you check the details of booking CW-9999?
- **Expectation**: Non-existent booking ID CW-9999 handled gracefully
- **Tools Called**: `['get_booking']`
- **Agent Answer**:
```text
Booking CW-9999 was not found in our system. Please double-check your booking ID. 

If you need further assistance, feel free to contact CityWheels staff:
- **Email:** help@citywheels.example
- **Phone:** +48 000 000 000 (every day during shop opening hours)
```

### Q12 (refusal) — PASSED
- **Question**: What is the capital of Australia?
- **Expectation**: Out-of-scope general knowledge question (must refuse and direct to staff)
- **Tools Called**: `[]`
- **Agent Answer**:
```text
I can only answer questions based on CityWheels policies and bookings. 

If you have any questions regarding CityWheels bike rentals, policies, or existing bookings, please let me know! For other inquiries, you can contact CityWheels staff directly:
- **Email:** `help@citywheels.example`
- **Phone:** `+48 000 000 000` (available every day during shop opening hours)
```

### Q13 (refusal) — PASSED
- **Question**: How do I bake a chocolate cake at home?
- **Expectation**: Out-of-scope cooking question (must refuse and direct to staff)
- **Tools Called**: `[]`
- **Agent Answer**:
```text
I can only answer questions based on CityWheels policies, bike rentals, and bookings. I am unable to answer off-topic questions or provide baking recipes. 

If you have any questions about CityWheels rentals or policies, please feel free to ask, or contact the CityWheels staff directly:
- **Email:** `help@citywheels.example`
- **Phone:** `+48 000 000 000` (every day while the shop is open)
```

### Q14 (escalation) — PASSED
- **Question**: There is a storm warning for my rental day. Will I get a full refund automatically?
- **Expectation**: Bad weather policy: staff decide each case, customer must contact staff
- **Tools Called**: `['search_policies']`
- **Agent Answer**:
```text
According to our policy, if the weather service issues a storm warning for the day of your rental, you may be eligible for a full refund. However, this is not automatic; our staff decides each case individually.

To request a refund due to a storm warning, please contact our staff directly:
- **Email:** help@citywheels.example
- **Phone:** +48 000 000 000 (available every day during shop hours)
```
