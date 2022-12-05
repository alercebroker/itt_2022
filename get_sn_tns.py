import http.client
import pandas as pd
from io import StringIO


class TNS:
    def __init__(self):
        self.tns_query_params = {
            "format": "csv",
            "num_page": 500,
            "groupid[]": 48,  # discovered by ZTF
            "classified_sne": 1,  # confirmed SNe
            "discovered_period_value": 10,
            "discovered_period_units": "years"
        }
        self.conn = http.client.HTTPSConnection("www.wis-tns.org")

    def _query_one_page(self, params_query):
        params = "&".join([f"{k}={params_query[k]}" for k in params_query])

        payload = ""

        self.conn.request("GET", f"/search?{params}", payload)

        res = self.conn.getresponse()
        data = res.read()

        text = data.decode("utf-8")
        df = pd.read_csv(StringIO(text))
        return df

    def _query_many_pages(self, params):
        pages = []
        page = 0
        try:
            while True:
                params["page"] = page
                temp_page = self._query_one_page(params)
                print(f"Page {page:3} -> {temp_page.shape[0]:4}")
                pages.append(temp_page)
                if temp_page.shape[0] < 500:
                    break
                page += 1
        except Exception as e:
            print(f"Error in page {page}: {e}")
        except KeyboardInterrupt as e:
            print(f"Stopped reader in {page}: {e}")
        df = pd.concat(pages)
        return df

    def get_confirmed_sn_from_ztf(self):
        tns_pages = self._query_many_pages(self.tns_query_params)
        return tns_pages


if __name__ == '__main__':
    tns = TNS()
    ztf_sn = tns.get_confirmed_sn_from_ztf()
    print(ztf_sn)
    ztf_sn.to_parquet('ztf_confirmed_sn.parquet')
