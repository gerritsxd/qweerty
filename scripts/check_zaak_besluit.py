"""Check Zaak-Besluit relationship in OData API."""
import requests

# Try Zaak with expand to Besluit
r = requests.get(
    "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Zaak?$top=1&$expand=Besluit&$format=json"
)
print("Zaak expand Besluit status:", r.status_code)
if r.status_code == 200:
    j = r.json()
    v = j.get("value", [])
    if v:
        print("Zaak keys:", list(v[0].keys()))
        if "Besluit" in v[0]:
            print("Besluit in response:", type(v[0]["Besluit"]))
