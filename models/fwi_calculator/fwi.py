from destinepyauth import get_token
import requests

HDA_STAC_ENDPOINT="https://hda.data.destination-earth.eu/stac/v2"
COLLECTION_ID = "EO.EUM.DAT.MSG.LSA-FRM"

def main():
    access_token = get_token("cacheb").access_token
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.post(HDA_STAC_ENDPOINT+"/search", headers=auth_headers, json={
        "collections": [COLLECTION_ID],
    })
    print(response)

if __name__ == "__main__":
    main()
