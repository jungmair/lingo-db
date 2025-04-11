set persist=1;
create table zillow
(
    title                  text,
    address                text,
    city                   text,
    state                  text,
    postal_code            text,
    price                  text,
    facts_and_features   text,
    real_estate_provider text,
    url text
);
copy zillow from 'zillow.csv' delimiter ',';
