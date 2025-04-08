# Marketing-Analytics

[Marketing Campaign - Customer Targeting Tool](https://marketing-decisions-mayank.streamlit.app)

**Problem Statement:**

Website XYZ, a music-listening social networking website, follows the "freemium" business model. The website offers basic services for free, and provides a
number of additional premium capabilities for a monthly subscription fee. We are interested in predicting which people would be likely to convert from free users to premium subscribers
in the next 6 month period, if they are targeted by our promotional campaign.

**Data Description:**

The labeled dataset contains 41,540 records (1540 adopters and 40,000 non-adopters), each record representing
a diferent user of the XYZ website who was targeted in the previous marketing campaign.
Each record is described with 25 attributes. Here is a brief description of the attributes (attribute name/type/explanation):

- adopter / binary (0 or 1) / whether a user became a subscriber within the 6 month
period after the marketing campaign (this is the outcome variable!)

- user id / integer / unique user id (obviously, this is not a predictive feature, just a
unique identier )

- age / integer / age in years

- male / integer (0 or 1) / 1 { male, 0 { female

- friend cnt / integer / numbers of friends that the current user has

- avg friend age / real / average age of friends (in years)

- avg friend male / real (between 0 and 1) / percentage of males among friends

- friend country cnt / integer / number of diferent countries among friends of the current
user

- subscriber friend cnt / integer / number of friends who are subscribers of the premium
service

- songsListened / integer / total number of tracks this user listened (or reported as
listened)

- lovedTracks / integer / total number of diferent songs that the user \liked"

- posts / integer / number of forum or discussion board posts made by the user

- playlists / integer / number of playlists created by the user

- shouts / integer / number of wall posts received by the user

- good country / integer (0 or 1) / country type of the user: 0 { countries where free
usage is more limited, 1 { less limited

- tenure / integer / number of months since the user has registered on the website.

- There are also a number of attributes with the following names: delta <attr-name >,
where <attr-name > is one of the attributes mentioned in the above list. Such at-
tributes refer not to the overall number, but the change to the corresponding number
over the 3-month period before the marketing campaign. For example, consider at-
tribute delta friend cnt. If, for some user, friend cnt = 50, and delta friend cnt = {5,
it means that the user had 50 friends at the time of the previous marketing campaign,
but this number reduced by 5 during the 3 months before the campaign (i.e., user had
55 friends 3 months ago).

**Ask:**

The general task is to build the best predictive model for the next marketing campaign,
i.e., for predicting likely adopters (that is, which current non-subscribers are likely to respond
to the marketing campaign and sign up for the premium service within 6 months after the
campaign).
