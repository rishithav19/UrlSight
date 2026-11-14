
from collections import defaultdict

from html_dom.dataset import load_split

ROOT = "../../datasets/phish360/Phish360"

records, skipped = load_split(ROOT, "trainval")

stats = defaultdict(lambda: defaultdict(int))

for record in records:

    label = record["label"]
    dom = record["dom_features"]

    stats[label]["samples"] += 1

    password = dom.get("password_input_count", 0) > 0
    external_action = dom.get("form_external_action_count", 0) > 0
    http_action = dom.get("form_http_action_count", 0) > 0
    https_action = dom.get("form_https_action_count", 0) > 0
    email = dom.get("email_input_count", 0) > 0
    get_form = dom.get("form_count_with_get", 0) > 0

    if password:
        stats[label]["password_pages"] += 1

    if password and external_action:
        stats[label]["password_external_action"] += 1

    if password and http_action:
        stats[label]["password_http_action"] += 1

    if password and https_action:
        stats[label]["password_https_action"] += 1

    if password and email:
        stats[label]["password_and_email"] += 1

    if not password and get_form:
        stats[label]["get_without_password"] += 1

    if external_action and not password:
        stats[label]["external_action_without_password"] += 1


print("\nFORM COMBINATION ANALYSIS")
print("Class 0 = Legitimate | Class 1 = Phishing")

for label in [0, 1]:

    print(f"\n{'LEGITIMATE' if label == 0 else 'PHISHING'}")

    total = stats[label]["samples"]

    for feature, count in stats[label].items():

        if feature == "samples":
            continue

        percentage = count / total * 100

        print(
            f"{feature}: {count}/{total} "
            f"({percentage:.2f}%)"
        )
