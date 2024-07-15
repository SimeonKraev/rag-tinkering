# GitHub Stargazers

A small repo for analysing the number and frequency of GitHib stars for a particular repo.

## Whats under the hood?

Under src/methods/funcs.py you will find...

**Stargazer** - somebody who stars a repository

The **GitHubStargazers** class is designed to interact with the GitHub API to fetch and analyze stargazer data for a specified repository. It provides functionalities to parse the repository URL, retrieve stargazer information, aggregate and convert this data into a pandas DataFrame, and finally, plot the number of stars the repository has received over time. Alternatively you can run it as a flask server as is or through Docker and make API calls to get the raw data. Below is a breakdown of its functionalities and methods:

+ **__init__(self, url, save_data=False)**: Initializes the class with the repository URL and a flag indicating whether to save the data to a CSV file. It sets up the necessary headers for GitHub API requests and specifies the columns of interest from the stargazer data.

+ **parse_url(self)**: Extracts the user and repository name from the provided URL and updates the URL to point directly to the stargazers endpoint of the GitHub API.

+ **get_stargazers(self)**: Retrieves the stargazers for the repository in a paginated manner, handling rate limits and potential errors. It accumulates all stargazer data into a list.

+ **aggregate_data(self)**: Processes the raw stargazer data to extract relevant information (as specified in self.usr_info) and aggregates it into a list of dictionaries, each representing a stargazer.

+ **data_to_df(self)**: Converts the aggregated stargazer data into a pandas DataFrame, sorts it by the date when the star was given, and optionally saves this data to a CSV file if self.save_data is True.

+ **plot_stars(self)**: Plots the number of stars the repository has received over time using matplotlib. It resamples the data to a yearly frequency and plots the result, showing the growth in the number of stars.

This class provides a comprehensive tool for analyzing the popularity and growth of GitHub repositories through their stargazers, leveraging data fetching, processing, and visualization capabilities.

## Endpoints

### Health Check Endpoint
- **Path**: `/health`
- **Method**: `GET`
- **Description**: This endpoint is used to check the health of the application. It can be used for monitoring and alerting purposes to ensure the application is running correctly.
- **Response**:
  - **Status Code**: `200 OK`
  - **Body**:
    ```json
    {
      "status": "healthy"
    }
    ```

### Star History Endpoint
- **Path**: `/star-history`
- **Method**: `GET`
- **Description**: Retrieves the star history of a specified GitHub repository. This endpoint requires a URL parameter that specifies the GitHub repository.
- **Query Parameters**:
  - **url**: The URL of the GitHub repository for which to retrieve the star history.
- **Response**:
  - **Status Code**: `200 OK`
  - **Body**:
    ```json
    {
      "content": [{"starred_at": date1, "id": 1337},
                  {"starred_at": date2, "id": 1338}]
    }
    ```

## Run it

`docker build -t stargazer .`

`docker run -p 5000:5000 stargazer`

alternatively

`pip install -r requirements.txt`

`python main.py https://github.com/repo-owner/repo-name`

## Test it via API call

`curl "http://localhost:5000/health"`

`curl "http://localhost:5000/star-history?url=https://github.com/repo_owner/repo_name"`