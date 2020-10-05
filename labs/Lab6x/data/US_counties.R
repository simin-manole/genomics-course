library(tidyverse)
library(maps)
library(mapdata)
library(lubridate)
library(viridis)
library(wesanderson)
library(RColorBrewer)

# Get and format the covid report data
report_09_26_2020 <- read_csv(url("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/09-26-2020.csv")) %>% 
  rename(Long = "Long_") %>% 
  unite(Key, Admin2, Province_State, sep = ".") %>% 
  group_by(Key) %>% 
  summarize(Confirmed = sum(Confirmed)) %>% 
  mutate(Key = tolower(Key))

# get and format the map data
us <- map_data("state")
counties <- map_data("county") %>% 
  unite(Key, subregion, region, sep = ".", remove = FALSE)

# Join the 2 tibbles
state_join <- left_join(counties, report_09_26_2020, by = c("Key"))

# sum(is.na(state_join$Confirmed))
#ggplot(data = us, mapping = aes(x = long, y = lat, group = group)) + 
#  coord_fixed(1.3) + 
#  # Add data layer
#  borders("state", colour = "black") +
#  geom_polygon(data = state_join, aes(fill = Confirmed)) +
#  scale_fill_gradientn(colors = brewer.pal(n = 5, name = "Blues"),
#                       breaks = c(1, 10, 100, 1000, 10000, 100000),
#                       trans = "log10", na.value = "White") +
#  ggtitle("Number of Confirmed Cases by US County, 9/26/2020") +
#  theme_bw() 