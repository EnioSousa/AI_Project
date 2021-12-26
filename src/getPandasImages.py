from requests import exceptions
import argparse
import requests
import cv2
import os


def getPandaImages(animal: str, path: str, total: int):
    # set your Microsoft Cognitive Services API key along with (1) the
    # maximum number of results for a given search and (2) the group size
    # for results (maximum of 50 per request)
    API_KEY = "e6c174a192334a169888079b95eeac29"
    MAX_RESULTS = 2500
    GROUP_SIZE = 50
    # set the endpoint API URL
    URL = "https://api.bing.microsoft.com/v7.0/images/search"

    # when attempting to download images from the web both the Python
    # programming language and the requests library have a number of
    # exceptions that can be thrown so let's build a list of them now
    # so we can filter on them
    EXCEPTIONS = set([IOError, FileNotFoundError,
                      exceptions.RequestException, exceptions.HTTPError,
                      exceptions.ConnectionError, exceptions.Timeout])

    # store the search term in a convenience variable then set the
    # headers and search parameters
    term = animal
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    params = {"q": term, "offset": 0, "count": GROUP_SIZE}
    # make the search
    print("[INFO] searching Bing API for '{}'".format(term))
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    # grab the results from the search, including the total number of
    # estimated results returned by the Bing API
    results = search.json()
    estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
    print("[INFO] {} total results for '{}'".format(estNumResults,
                                                    term))

    # loop over the estimated number of results in `GROUP_SIZE` groups
    for offset in range(0, estNumResults, GROUP_SIZE):
        # update the search parameters using the current offset, then
        # make the request to fetch the results
        print("[INFO] making request for group {}-{} of {}...".format(
            offset, offset + GROUP_SIZE, estNumResults))
        params["offset"] = offset
        search = requests.get(URL, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()
        print("[INFO] saving images for group {}-{} of {}...".format(
            offset, offset + GROUP_SIZE, estNumResults))

        # loop over the results
        for v in results["value"]:
            # try to download the image
            try:
                # make a request to download the image
                print("[INFO] fetching: {}".format(v["contentUrl"]))
                r = requests.get(v["contentUrl"], timeout=30)
                # build the path to the output image
                ext = v["contentUrl"][v["contentUrl"].rfind("."):]

                if ( ext == ".jpg"):
                    p = os.path.sep.join([path, animal + "{}{}".format(
                        str(total).zfill(8), ext)])
                    # write the image to disk
                    f = open(p, "wb")
                    f.write(r.content)
                    f.close()

                else:
                    print("[INFO] Ignore image: Not in jpg format")
                    continue
            # catch any errors that would not unable us to download the
            # image
            except Exception as e:
                # check to see if our exception is in our list of
                # exceptions to check for
                if type(e) in EXCEPTIONS:
                    print("[INFO] skipping: {}".format(v["contentUrl"]))
                    continue

    # try to load the image from disk
            image = cv2.imread(p)
            # if the image is `None` then we could not properly load the
            # image from disk (so it should be ignored)
            if image is None:
                print("[INFO] deleting: {}".format(p))
                if ( os.path.exists(p)):
                    os.remove(p)
                continue
            # update the counter
            total += 1
