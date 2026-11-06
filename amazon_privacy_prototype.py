"""Small teaching prototype for the PSP301 Amazon privacy report.

Course concepts demonstrated in simple test cases:
- Access control: order ownership and denial of cross-account requests.
- Data minimisation: return only the order fields needed for the response.
- Authentication/risk response: ask for extra verification for a new device.
- Explainability: show reasons for rule-based risk decisions and recommendations.

Uses synthetic records only. It is not connected to Amazon or real customer data.
The checks show that these simplified examples work as written; they do not prove
production security, real-world fraud-detection performance, or Amazon integration.
"""

ORDERS = {
    "ORD-100": {
        "owner_id": "user-01",
        "product": "Wireless Mouse",
        "status": "Shipped",
        "email": "user01@example.test",
        "delivery_address": "10 Example Road",
    },
    "ORD-200": {
        "owner_id": "user-02",
        "product": "Notebook",
        "status": "Processing",
        "email": "user02@example.test",
        "delivery_address": "20 Example Road",
    },
}

PUBLIC_ORDER_FIELDS = ("order_id", "product", "status")


def get_order(user_id, order_id):
    """Return an allowlisted order view only after verifying ownership."""
    order = ORDERS.get(order_id)
    if order is None or order["owner_id"] != user_id:
        raise PermissionError("Order access denied: ownership check failed")
    return {key: order[key] for key in PUBLIC_ORDER_FIELDS if key in order} | {"order_id": order_id}


def assess_transaction(event):
    """Explainable rule-based baseline; not a trained ML fraud model."""
    score = 0
    reasons = []
    if event.get("new_device"):
        score += 1
        reasons.append("new device")
    if event.get("refunds_24h", 0) >= 3:
        score += 2
        reasons.append("repeated refund requests")
    if event.get("payout_change"):
        score += 3
        reasons.append("seller payout details changed")

    if score >= 3:
        action = "temporary_hold_for_review"
    elif score >= 1:
        action = "step_up_verification"
    else:
        action = "allow"
    return {"action": action, "score": score, "reasons": reasons}


def recommend(pseudonymous_user_id, interactions, catalogue):
    """Rank unseen products from category interaction counts; no identity fields accepted."""
    category_scores = {}
    seen = set()
    for row in interactions:
        if row["user_id"] != pseudonymous_user_id:
            continue
        category = row["category"]
        category_scores[category] = category_scores.get(category, 0) + row.get("weight", 1)
        if row.get("product_id"):
            seen.add(row["product_id"])
    ranked = [p for p in catalogue if p["id"] not in seen]
    ranked.sort(key=lambda p: category_scores.get(p["category"], 0), reverse=True)
    return [{"product_id": p["id"], "category": p["category"],
             "reason": f"matches interacted category: {p['category']}"}
            for p in ranked if category_scores.get(p["category"], 0) > 0][:3]


def run_checks():
    results = []
    # 1. Owner can view own order.
    own = get_order("user-01", "ORD-100")
    assert own["status"] == "Shipped"
    results.append("PASS: owner can view their order")

    # 2. Another account cannot view the order.
    try:
        get_order("user-02", "ORD-100")
        raise AssertionError("cross-account request unexpectedly allowed")
    except PermissionError:
        results.append("PASS: cross-account order request denied")

    # 3. Response uses allowlisted fields only.
    assert set(own.keys()) == {"order_id", "product", "status"}
    results.append("PASS: email and delivery address excluded from order response")

    # 4. Low-risk activity proceeds; high-risk activity has reason labels.
    low = assess_transaction({"new_device": False, "refunds_24h": 0, "payout_change": False})
    high = assess_transaction({"new_device": True, "refunds_24h": 3, "payout_change": True})
    assert low["action"] == "allow"
    assert high["action"] == "temporary_hold_for_review" and high["reasons"]
    results.append("PASS: risk decisions differ by event and include reason labels")

    # 5. New device alone triggers verification rather than an automatic block.
    new_device = assess_transaction({"new_device": True, "refunds_24h": 0, "payout_change": False})
    assert new_device["action"] == "step_up_verification"
    results.append("PASS: a new device alone triggers verification, not a hold")

    # 6. Recommender works from pseudonymous interactions and explains ranking.
    interactions = [
        {"user_id": "p-abc", "product_id": "P1", "category": "books", "weight": 2},
        {"user_id": "p-abc", "product_id": "P2", "category": "books", "weight": 1},
    ]
    catalogue = [
        {"id": "P1", "category": "books"},
        {"id": "P3", "category": "books"},
        {"id": "P4", "category": "electronics"},
    ]
    recs = recommend("p-abc", interactions, catalogue)
    assert recs and recs[0]["product_id"] == "P3" and "reason" in recs[0]
    results.append("PASS: recommendations use pseudonymous interactions and return an explanation")
    return results


if __name__ == "__main__":
    for result in run_checks():
        print(result)
    print(f"\n{len(run_checks())} checks passed.")
