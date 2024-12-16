import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


df = pd.read_csv('imdb_top_1000.csv')

# Data Cleaning

print("Rows before cleaning:", len(df))
df.dropna(subset=['IMDB_Rating', 'Gross', 'Genre', 'Released_Year'], inplace=True)
print("Rows after cleaning:", len(df))
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
df = df.sort_values(by='Released_Year')
df = df[df['Released_Year'].between(1900, 2024)]
df['Gross'] = df['Gross'].replace(r'[\$,]', '', regex=True).astype(float)
df['Decade'] = (df['Released_Year'] // 10) * 10
df['Primary_Genre'] = df['Genre'].apply(lambda x: x.split(',')[0].strip())


# Code for Top 10 Best Performing Actors over Century:
actor_df = df.melt(id_vars=['Released_Year', 'IMDB_Rating', 'No_of_Votes', 'Series_Title', 'Runtime', 'Genre', 'Gross', 'Overview', 'Director'],
                    value_vars=['Star1', 'Star2', 'Star3', 'Star4'],
                    var_name='Star_Role', value_name='Actor')
actor_df.dropna(subset=['Actor'], inplace=True)


def calculate_cumulative_data(actor_df, year):
    yearly_data = actor_df[actor_df['Released_Year'] <= year]
    cumulative_df = (
        yearly_data
        .groupby(['Actor', 'Series_Title'], as_index=False)
        .agg({
            'No_of_Votes': 'sum',         # Total votes per movie
            'IMDB_Rating': 'mean',       # IMDb Rating per movie
            'Released_Year': 'max',
            'Director': 'first',
            'Runtime': 'first',
            'Genre': 'first',
            'Gross': 'first',
            'Overview': 'first'
        })
        .rename(columns={'No_of_Votes': 'Movie_Votes'})
    )

    cumulative_df = cumulative_df.sort_values(by=['Actor', 'Released_Year'], ascending=[True, True])

    cumulative_df['Cumulative_Movie_Count'] = cumulative_df.groupby('Actor').cumcount() + 1
    cumulative_df['Cumulative_Votes'] = cumulative_df.groupby('Actor')['Movie_Votes'].cumsum()
    cumulative_df['Cumulative_Rating'] = cumulative_df.groupby('Actor')['IMDB_Rating'].expanding().mean().reset_index(level=0, drop=True)

    return cumulative_df


actor_df['Released_Year'] = actor_df['Released_Year'].astype(int)

animated_data = []
for year in range(actor_df['Released_Year'].min(), actor_df['Released_Year'].max() + 1):
    yearly_cumulative_df = calculate_cumulative_data(actor_df, year)
    actor_order = (
        yearly_cumulative_df.groupby('Actor', as_index=False)
        .agg({
            'Cumulative_Movie_Count': 'max',
            'Cumulative_Votes': 'max'
        })
        .sort_values(by=['Cumulative_Votes', 'Cumulative_Movie_Count'], ascending=[False, False])
    )['Actor']

    top_actors = actor_order.head(10)
    filtered_data = yearly_cumulative_df[yearly_cumulative_df['Actor'].isin(top_actors)].copy()
    filtered_data['Year'] = year

    # sort movies chronologically within each actor
    filtered_data = filtered_data.sort_values(by=['Actor', 'Released_Year'], ascending=[True, True])

    filtered_data['Movie_Order'] = filtered_data.groupby('Actor').cumcount() + 1
    filtered_data['Actor_Summary'] = None
    last_bar_indices = filtered_data.groupby('Actor')['Movie_Order'].idxmax()
    filtered_data.loc[last_bar_indices, 'Actor_Summary'] = filtered_data.apply(
        lambda row: f"{row['Cumulative_Movie_Count']} movies, Avg IMDb: {row['Cumulative_Rating']:.1f}, Total Votes: {row['Cumulative_Votes']:,}",
        axis=1
    )
    filtered_data['Actor'] = pd.Categorical(filtered_data['Actor'], categories=actor_order, ordered=True)
    filtered_data = filtered_data.sort_values(by=['Actor', 'Released_Year'])
    animated_data.append(filtered_data)

final_data = pd.concat(animated_data, ignore_index=True)

final_data['Hover_Text'] = (
    "<br>" + "<b> Movie Title:</b> " + final_data['Series_Title'] + "<br>" +
    "<b>Overview:</b> " + final_data['Overview'].fillna("N/A") + "<br>" +
    "<b>Director:</b> " + final_data['Director'].fillna("N/A") + "<br>" +
    "<b>Year of Release:</b> " + final_data['Released_Year'].astype(str) + "<br>" +
    "<b>Runtime:</b> " + final_data['Runtime'].fillna("N/A") + "<br>" +
    "<b>Genre:</b> " + final_data['Genre'].fillna("N/A") + "<br>" +
    "<b>Gross Revenue:</b> €" + final_data['Gross'].fillna("N/A").map('{:,}'.format)
)

fig = px.bar(
    final_data,
    x="Movie_Votes",
    y="Actor",
    color="IMDB_Rating",
    animation_frame="Year",
    hover_data={"Hover_Text": True},
    text=final_data['Actor_Summary'],
    title=f"Top 10 Best Performing Actors over the Century (Based on Movie Votes)",
    labels={"Movie_Votes": "Movie Votes", "Actor": "Actor"}
)

fig.update_traces(
    hovertemplate="%{customdata[0]}",
    customdata=final_data[["Hover_Text"]],
    hoverlabel=dict(
        bgcolor="lightyellow",
        # font_size=12,
        # font_family="Arial"
    )
)

fig.update_layout(
    height=800,
    width=1200,
    margin=dict(l=50, r=50, t=70, b=50),
    xaxis=dict(title="Cumulative Movie Votes", range=[0, 16000000], showgrid=False),
    yaxis=dict(autorange="reversed", title="Actor"),
    coloraxis=dict(
        cmin=7, cmax=10,
        colorscale='RdYlGn',
        colorbar=dict(title="IMDb Rating")
    ),
    font=dict(family="Arial", size=11.5),
    sliders=[{
        "pad": {"b": 10},
        "len": 0.8,
        "xanchor": "center", "x": 0.5
    }],
    updatemenus=[{
        "buttons": [
            {"method": "animate", 
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}], 
            "label": "Play"},
            {"method": "animate", 
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], 
            "label": "Pause"}
        ],
        "x": 0.57, "y": -0.15
    }]
)


# Code for Top 10 Best Performing Directors over Century:
def calculate_cumulative_data(director_df, year):
    yearly_data = director_df[director_df['Released_Year'] <= year]
    cumulative_df = (
        yearly_data
        .groupby(['Director', 'Series_Title'], as_index=False)
        .agg({
            'No_of_Votes': 'sum',         # Total votes per movie
            'IMDB_Rating': 'mean',       # IMDb Rating per movie
            'Released_Year': 'max',
            'Director': 'first',
            'Runtime': 'first',
            'Genre': 'first',
            'Gross': 'first',
            'Overview': 'first'
        })
        .rename(columns={'No_of_Votes': 'Movie_Votes'})
    )

    cumulative_df = cumulative_df.sort_values(by=['Director', 'Released_Year'], ascending=[True, True])
    cumulative_df['Cumulative_Movie_Count'] = cumulative_df.groupby('Director').cumcount() + 1
    cumulative_df['Cumulative_Votes'] = cumulative_df.groupby('Director')['Movie_Votes'].cumsum()
    cumulative_df['Cumulative_Rating'] = cumulative_df.groupby('Director')['IMDB_Rating'].expanding().mean().reset_index(level=0, drop=True)

    return cumulative_df

director_df = df
director_df.dropna(subset=['Director'], inplace=True)
director_df['Released_Year'] = director_df['Released_Year'].astype(int)

animated_data = []
for year in range(director_df['Released_Year'].min(), director_df['Released_Year'].max() + 1):
    yearly_cumulative_df = calculate_cumulative_data(director_df, year)
    director_order = (
        yearly_cumulative_df.groupby('Director', as_index=False)
        .agg({
            'Cumulative_Movie_Count': 'max',
            'Cumulative_Votes': 'max',
            'Cumulative_Rating': 'mean'
        })
        .sort_values(by=["Cumulative_Votes", 'Cumulative_Movie_Count'], ascending=[False, False])
    )['Director']

    top_directors = director_order.head(10)
    filtered_data = yearly_cumulative_df[yearly_cumulative_df['Director'].isin(top_directors)].copy()
    filtered_data['Year'] = year

    filtered_data = filtered_data.sort_values(by=['Director', 'Released_Year'], ascending=[True, True])

    filtered_data['Movie_Order'] = filtered_data.groupby('Director').cumcount() + 1

    filtered_data['Director_Summary'] = None
    last_bar_indices = filtered_data.groupby('Director')['Movie_Order'].idxmax()
    filtered_data.loc[last_bar_indices, 'Director_Summary'] = filtered_data.apply(
        lambda row: f"{row['Cumulative_Movie_Count']} movies, Avg IMDb: {row['Cumulative_Rating']:.1f}, Total Votes: {row['Cumulative_Votes']:,}",
        axis=1
    )

    filtered_data['Director'] = pd.Categorical(filtered_data['Director'], categories=director_order, ordered=True)
    filtered_data = filtered_data.sort_values(by=['Director', 'Released_Year'])
    animated_data.append(filtered_data)

final_data = pd.concat(animated_data, ignore_index=True)
final_data['Hover_Text'] = (
    "<br>" + "<b> Movie Title:</b> " + final_data['Series_Title'] + "<br>" +
    "<b>Overview:</b> " + final_data['Overview'].fillna("N/A") + "<br>" +
    "<b>Year of Release:</b> " + final_data['Released_Year'].astype(str) + "<br>" +
    "<b>Runtime:</b> " + final_data['Runtime'].fillna("N/A") + "<br>" +
    "<b>Genre:</b> " + final_data['Genre'].fillna("N/A") + "<br>" +
    "<b>Gross Revenue:</b> €" + final_data['Gross'].fillna("N/A").map('{:,}'.format)
)

fig_directors = px.bar(
    final_data,
    x="Movie_Votes",
    y="Director",
    color="IMDB_Rating",
    animation_frame="Year",
    hover_data={"Hover_Text": True},
    text=final_data['Director_Summary'],
    title=f"Top 10 Best Performing Directors Over the Century (Based on Movie Votes)",
    labels={"Movie_Votes": "Movie Votes", "Director": "Director"}
)

fig_directors.update_traces(
    hovertemplate="%{customdata[0]}",
    customdata=final_data[["Hover_Text"]],
    hoverlabel=dict(
        bgcolor="lightyellow",
        # font_size=12,
        # font_family="Arial"
    )
)


fig_directors.update_layout(
    height=800,
    width=1200,
    margin=dict(l=50, r=50, t=70, b=50),
    xaxis=dict(title="Cumulative Movie Votes", range=[0, 16000000], showgrid=False),
    yaxis=dict(autorange="reversed", title="Director"),
    coloraxis=dict(
        cmin=7, cmax=10,
        colorscale='RdYlGn',
        colorbar=dict(title="IMDb Rating")
    ),
    font=dict(family="Arial", size=11.5),
    sliders=[{
        "pad": {"b": 10},
        "len": 0.8,
        "xanchor": "center", "x": 0.5
    }],
    updatemenus=[{
        "buttons": [
            {"method": "animate", 
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}], 
            "label": "Play"},
            {"method": "animate", 
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], 
            "label": "Pause"}
        ],
        "x": 0.57, "y": -0.15
    }]
)

# fig_directors.show()


# Code for Scatterplot of Votes vs IMDB rating with Genre and Gross Revenue
top_movies_by_year = df.sort_values(['Released_Year', 'No_of_Votes'], ascending=[True, False])
top_movies_by_year = top_movies_by_year.groupby('Released_Year').head(10).reset_index()

top_movies_by_year['Genre'] = top_movies_by_year['Genre'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else 'Unknown')
fig_movies = px.scatter(
    top_movies_by_year,
    x='IMDB_Rating',
    y='No_of_Votes',
    size='Gross',
    color='Genre',
    hover_data={
        'Series_Title': True,
        'Overview': True,
        'Runtime': True,
        'Released_Year': True,
        'Gross': True,
        'IMDB_Rating': True,
        'Genre': True
    },
    title="Votes vs IMDB rating with Genre and Gross Revenue",
    color_discrete_sequence=px.colors.qualitative.Set1
)

fig_movies.update_traces(
    hovertemplate=(
        "<b>Title:</b> %{customdata[0]}<br>"
        "<b>Overview:</b> %{customdata[1]}<br>"
        "<b>Runtime:</b> %{customdata[2]}<br>"
        "<b>Year:</b> %{customdata[3]}<br>"
        "<b>Votes:</b> %{y:,}<br>"
        "<b>IMDb Rating:</b> %{x}<br>"
        "<b>Gross:</b> €%{customdata[4]:,}<br>"
        "<b>Genre:</b> %{customdata[5]}"
    )
)

fig_movies.update_layout(
    height=600,
    width=1300,
    xaxis=dict(title="IMDb Rating"),
    yaxis=dict(title="Total Votes", showgrid=True),
    margin=dict(l=50, r=50, t=70, b=50)
)
# fig_movies.update_layout(margin=dict(r=120))







# fig_movies.show()


# Code for Scatterplot of Runtime vs. Votes with Revenue Encoding
df['Runtime'] = df['Runtime'].str.extract('(\d+)').astype(float)
df['Runtime'] = df['Runtime'].fillna(0)

fig_runtime_votes = px.scatter(
    df,
    x='Runtime',
    y='No_of_Votes',
    size='Gross', # Size represents Gross Revenue
    color='IMDB_Rating',
    hover_data={
        'Series_Title': True,
        'Overview': True,
        'Genre': True,
        'Released_Year': True,
        'Runtime': True,
        'No_of_Votes': True,
        'Gross': True,
        'IMDB_Rating': True,
    },
    title="Runtime vs. Votes with Gross Revenue and IMDb Rating",
    color_continuous_scale="RdYlGn"
)
fig_runtime_votes.update_traces(
    hovertemplate=(
        "<b>Title:</b> %{customdata[0]}<br>"
        "<b>Overview:</b> %{customdata[1]}<br>"
        "<b>Genre:</b>%{customdata[2]}<br>"
        "<b>Year:</b> %{customdata[3]}<br>"
        "<b>Runtime:</b><b>%{x} mins</b><br>"
        "<b>Votes:</b><b>%{y:,}</b><br>"
        "<b>Gross Revenue:</b> <b>€%{customdata[4]:,}</b><br>"
        "<b>IMDb Rating:</b> <b>%{customdata[5]}</b>"
    )
)
fig_runtime_votes.update_layout(
    height=600,
    width=1100,
    xaxis=dict(
        title="Movie Runtime (minutes)",
        tickmode='linear',
        dtick=15,
        range=[60, df['Runtime'].max() + 10],
        showgrid=True
    ),
    coloraxis=dict(
        cmin=7, cmax=10,
        colorscale='RdYlGn',
        colorbar=dict(title="IMDb Rating")
    ),
    yaxis=dict(title="Total Votes", showgrid=True),
    coloraxis_colorbar=dict(title="IMDb Rating"),
    margin=dict(l=50, r=50, t=70, b=50)
)



# Code for Boxplot of distribution of Gross Revenue by Movie Certificate
fig_cert_revenue = px.box(
    df,
    x='Certificate',
    y='Gross',
    title="Distribution of Gross Revenue by Movie Certificate (Boxplot)",
    labels={"Gross": "Gross Revenue", "Certificate": "Movie Certificate"}
)
fig_cert_revenue.update_layout(
    height=600,
    width=900,
    xaxis=dict(title="Certificate"),
    yaxis=dict(title="Gross Revenue (€)", showgrid=True, tickformat=".2s"),  # Format revenue
    margin=dict(l=50, r=50, t=50, b=50)
)
# fig_cert_revenue.show()




# figures in dark mode
for fig_obj in [fig_movies, fig_runtime_votes, fig_cert_revenue, fig, fig_directors]:
    fig_obj.update_layout(
        paper_bgcolor='#000000',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#FFFFFF', family='Arial, sans-serif'),
        xaxis=dict(gridcolor='#444444'),
        yaxis=dict(gridcolor='#444444')
    )

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div([
            # IMDb Logo
            html.Img(
                src="/assets/imdb_logo.png",
                style={
                    'height': '80px',
                    'margin-left': '10px',
                    'margin-right': '10px',
                    'margin-top': '5px',
                    'float': 'left'
                }
            ),
            html.H1(
                "Top 1000 Movies Data Dashboard",
                style={
                    'textAlign': 'center',
                    'color': '#FFD700',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '30px',
                    'fontWeight': 'bold',
                    'textShadow': '1px 1px 2px #000000',
                    'lineHeight': '80px',
                    'letterSpacing': '0.5px',
                    'margin': '0',
                }
            )
        ], style={
            'backgroundColor': '#000000',
            'borderBottom': '3px solid #FFD700',
            'height': '100px',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'padding-right': '20px'
        })
    ]),
    html.Div([
        html.Div([
            dcc.Graph(
                figure=fig_movies,
                style={"height": "600px", "width": "30%", "margin-right": "2%"}
            ),
            dcc.Graph(
                figure=fig_runtime_votes,
                style={"height": "600px", "width": "30%", "margin-left": "8%"}
            ),
            dcc.Graph(
                figure=fig_cert_revenue,
                style={"height": "600px", "width": "30%", "margin-left": "3%", "margin-right": "10%"}
            )
        ], style={
            "display": "flex",
            "justify-content": "space-between",
            "align-items": "flex-start",
            "margin-bottom": "20px"
        }),
        html.Div([
            dcc.Graph(
                figure=fig,
                style={"height": "600px", "width": "45%", "margin-left": "10%"}
            ),
            dcc.Graph(
                figure=fig_directors,
                style={"height": "600px", "width": "45%", "margin-right": "10%"}
            )
        ], style={
            "display": "flex",
            "justify-content": "center",
            "align-items": "flex-start",
            "margin-top": "20px",
            "gap": "40px"
        })
    ], style={
        "backgroundColor": "#000000",
        "padding-bottom": "20px"
    })
], style={
    "backgroundColor": "#000000",
    "min-height": "100vh"
})

if __name__ == '__main__':
    app.run_server(debug=True)
