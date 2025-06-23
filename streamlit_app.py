import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="NYT Film Ballots Explorer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to generate movie image URL
def get_movie_image_url(nyt_id):
    """Generate NYT image URL from nyt_id"""
    if pd.notna(nyt_id) and nyt_id.strip():
        return f"https://static01.nyt.com/newsgraphics/movie-survey-2025/img/300/{nyt_id}.jpg"
    return None

# Load data
@st.cache_data
def load_data():
    """Load all data files"""
    try:
        with open('nyt_film_data_stats.json', 'r') as f:
            stats = json.load(f)
        
        people_df = pd.read_csv('nyt_film_data_people.csv')
        movies_df = pd.read_csv('nyt_film_data_movies.csv')
        metadata_df = pd.read_csv('nyt_film_data_metadata.csv')
        
        return stats, people_df, movies_df, metadata_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Load data
stats, people_df, movies_df, metadata_df = load_data()

if stats is None:
    st.error("Could not load data files. Please ensure all CSV and JSON files are in the same directory.")
    st.stop()

# Sidebar navigation
st.sidebar.title("🎬 NYT Film Ballots Explorer")
st.sidebar.markdown("Navigate through different views of the film ballots data")

page = st.sidebar.selectbox(
    "Choose a page:",
    ["📊 Overview", "🎯 Top 100", "👥 People & Categories", "📈 Trends & Analysis", "🔍 Individual Ballots", "📊 Interactive Explorer"]
)

# Main title
st.title("🎬 NYT Film Ballots: Best Movies of the 21st Century")
st.markdown("*Analysis of 119 film industry professionals' movie preferences*")

# Overview Page
if page == "📊 Overview":
    st.header("📊 High-Level Statistics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Participants",
            value=stats['total_people'],
            help="Film industry professionals who participated"
        )
    
    with col2:
        st.metric(
            label="Total Movie Selections",
            value=f"{stats['total_movie_selections']:,}",
            help="Total number of movies selected across all ballots"
        )
    
    with col3:
        st.metric(
            label="Unique Movies",
            value=stats['unique_movies'],
            help="Number of different movies mentioned"
        )
    
    with col4:
        st.metric(
            label="Avg Movies per Person",
            value=f"{stats['average_movies_per_person']:.1f}",
            help="Average number of movies per ballot"
        )
    
    st.divider()
    
    # Survey info
    st.subheader("📰 Survey Information")
    survey_info = stats['survey_info']
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Article:** {survey_info['headline']}")
        st.info(f"**Published:** {survey_info['first_published']}")
        st.info(f"**Last Modified:** {survey_info['last_modified']}")
    
    with col2:
        # Category breakdown pie chart
        categories = [item[0] for item in stats['category_breakdown']]
        counts = [item[1] for item in stats['category_breakdown']]
        
        fig_pie = px.pie(
            values=counts,
            names=categories,
            title="Participants by Professional Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.divider()
    
    # Quick insights
    st.subheader("🔍 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Popular Movies:**")
        for i, (movie, count) in enumerate(stats['most_popular_movies'][:5], 1):
            st.write(f"{i}. **{movie}** - {count} selections")
    
    with col2:
        st.markdown("**Most Popular Years:**")
        for i, (year, count) in enumerate(stats['most_popular_years'][:5], 1):
            st.write(f"{i}. **{year}** - {count} movies selected")

# Top 100 Page
elif page == "🎯 Top 100":
    st.header("🎯 Top 100 Movies")
    st.markdown("*Comprehensive ranking of the most selected movies from all ballots*")
    
    # Calculate movie rankings
    movie_counts = movies_df['title'].value_counts()
    top_100 = movie_counts.head(100)
    
    # Controls section - centralized on same row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ranking_method = st.selectbox(
            "Select ranking method:",
            ["By Selection Count", "By Weighted Score (Higher ranks worth more)", "By Average Rank Position"]
        )
        
        # Show explanation for selected ranking method
        if ranking_method == "By Selection Count":
            st.info("Movies ranked by total number of selections across all ballots")
        elif ranking_method == "By Weighted Score (Higher ranks worth more)":
            st.info("Movies ranked by weighted score (higher rankings on individual ballots worth more points)")
        else:  # By Average Rank Position
            st.info("Movies ranked by average ranking position (lower average rank = better, minimum 2 selections)")
    
    with col2:
        display_format = st.radio(
            "Display format:",
            ["Gallery View", "List View", "Data Table"],
            horizontal=False
        )
    
    # Calculate rankings based on selected method
    if ranking_method == "By Selection Count":
        ranked_movies = movie_counts.head(100)
        
    elif ranking_method == "By Weighted Score (Higher ranks worth more)":
        # Calculate weighted scores (rank 1 = 10 points, rank 2 = 9 points, etc.)
        movie_scores = {}
        for _, movie in movies_df.iterrows():
            title = movie['title']
            # Higher ranks get more points (rank 1 = 10, rank 2 = 9, etc.)
            score = max(0, 11 - movie['rank'])  # Assuming max rank is 10
            if title in movie_scores:
                movie_scores[title] += score
            else:
                movie_scores[title] = score
        
        ranked_movies = pd.Series(movie_scores).sort_values(ascending=False).head(100)
        
    else:  # By Average Rank Position
        # Calculate average rank for movies with multiple selections
        avg_ranks = movies_df.groupby('title')['rank'].agg(['mean', 'count']).sort_values('mean')
        # Only include movies with at least 2 selections for meaningful averages
        avg_ranks = avg_ranks[avg_ranks['count'] >= 2]
        ranked_movies = avg_ranks.head(100)['mean']
    
    st.divider()
    
    if display_format == "Gallery View":
        # Gallery view with images
        st.subheader("🎬 Top 100 Movies Gallery")
        
        # Show in batches of 20
        batch_size = 20
        batch_selector = st.selectbox(
            "Select batch:",
            [f"Movies {i+1}-{min(i+batch_size, len(ranked_movies))}" for i in range(0, len(ranked_movies), batch_size)]
        )
        
        start_idx = int(batch_selector.split('-')[0].split()[-1]) - 1
        end_idx = min(start_idx + batch_size, len(ranked_movies))
        
        # Create 2x10 grid layout (10 columns per row)
        for row_start in range(start_idx, end_idx, 10):
            cols = st.columns(10)
            row_end = min(row_start + 10, end_idx)
            
            for idx in range(row_start, row_end):
                col_index = (idx - row_start) % 10
                movie_title = ranked_movies.index[idx]
                
                # Get movie details
                movie_info = movies_df[movies_df['title'] == movie_title].iloc[0]
                image_url = get_movie_image_url(movie_info['nyt_id'])
                
                with cols[col_index]:
                    if image_url:
                        st.image(image_url, caption=f"#{idx+1} {movie_title}", use_container_width=True)
                    else:
                        st.write(f"#{idx+1} **{movie_title}**")
                    
                    if ranking_method == "By Selection Count":
                        st.write(f"🗳️ **{ranked_movies.iloc[idx]} selections**")
                    elif ranking_method == "By Weighted Score (Higher ranks worth more)":
                        st.write(f"⭐ **{ranked_movies.iloc[idx]:.1f} points**")
                    else:
                        st.write(f"📊 **Avg rank: {ranked_movies.iloc[idx]:.1f}**")
            
            # Add spacing between rows
            if row_end < end_idx:
                st.write("")
    
    elif display_format == "List View":
        # Simple list view
        st.subheader("📋 Top 100 Movies List")
        
        # Split into two columns for better readability
        col1, col2 = st.columns(2)
        
        half_point = len(ranked_movies) // 2
        
        with col1:
            st.markdown("**Movies 1-50:**")
            for i in range(half_point):
                movie_title = ranked_movies.index[i]
                if ranking_method == "By Selection Count":
                    st.write(f"{i+1}. **{movie_title}** ({ranked_movies.iloc[i]} selections)")
                elif ranking_method == "By Weighted Score (Higher ranks worth more)":
                    st.write(f"{i+1}. **{movie_title}** ({ranked_movies.iloc[i]:.1f} points)")
                else:
                    st.write(f"{i+1}. **{movie_title}** (avg rank: {ranked_movies.iloc[i]:.1f})")
        
        with col2:
            st.markdown("**Movies 51-100:**")
            for i in range(half_point, len(ranked_movies)):
                movie_title = ranked_movies.index[i]
                if ranking_method == "By Selection Count":
                    st.write(f"{i+1}. **{movie_title}** ({ranked_movies.iloc[i]} selections)")
                elif ranking_method == "By Weighted Score (Higher ranks worth more)":
                    st.write(f"{i+1}. **{movie_title}** ({ranked_movies.iloc[i]:.1f} points)")
                else:
                    st.write(f"{i+1}. **{movie_title}** (avg rank: {ranked_movies.iloc[i]:.1f})")
    
    else:  # Data Table
        # Comprehensive data table
        st.subheader("📊 Top 100 Movies - Detailed Data")
        
        # Create comprehensive data
        detailed_data = []
        for i, movie_title in enumerate(ranked_movies.index):
            movie_data = movies_df[movies_df['title'] == movie_title]
            
            # Basic info
            year = movie_data['year'].iloc[0]
            selections = len(movie_data)
            avg_rank = movie_data['rank'].mean()
            
            # Calculate weighted score
            weighted_score = sum(max(0, 11 - rank) for rank in movie_data['rank'])
            
            # Get review URL
            review_url = movie_data['review_url'].iloc[0] if pd.notna(movie_data['review_url'].iloc[0]) else ""
            
            detailed_data.append({
                'Rank': i + 1,
                'Movie': movie_title,
                'Year': int(year),
                'Selections': selections,
                'Avg Rank': f"{avg_rank:.1f}",
                'Weighted Score': f"{weighted_score:.1f}",
                'Has Review': "Yes" if review_url else "No"
            })
        
        df_display = pd.DataFrame(detailed_data)
        st.dataframe(df_display, use_container_width=True)
        
        # Add download button
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Top 100 as CSV",
            data=csv,
            file_name="nyt_top_100_movies.csv",
            mime="text/csv"
        )
    
    st.divider()
    
    # Summary statistics
    st.subheader("📈 Top 100 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_selections = sum(movies_df[movies_df['title'].isin(ranked_movies.index)]['title'].value_counts())
        st.metric("Total Selections", f"{total_selections:,}")
    
    with col2:
        avg_year = movies_df[movies_df['title'].isin(ranked_movies.index)]['year'].mean()
        st.metric("Average Year", f"{avg_year:.0f}")
    
    with col3:
        year_range = movies_df[movies_df['title'].isin(ranked_movies.index)]['year']
        st.metric("Year Range", f"{int(year_range.min())}-{int(year_range.max())}")
    
    with col4:
        # Movies with reviews
        movies_with_reviews = movies_df[movies_df['title'].isin(ranked_movies.index)]['review_url'].notna().sum()
        st.metric("Movies with NYT Reviews", movies_with_reviews)
    
    

# People & Categories Page
elif page == "👥 People & Categories":
    st.header("👥 People & Professional Categories")
    
    # Category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        categories = [item[0] for item in stats['category_breakdown']]
        cat_counts = [item[1] for item in stats['category_breakdown']]
        
        fig_cat = px.bar(
            x=cat_counts,
            y=categories,
            orientation='h',
            title="Participants by Category",
            labels={'x': 'Number of Participants', 'y': 'Category'},
            color=cat_counts,
            color_continuous_scale='blues'
        )
        fig_cat.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Additional category insights
        st.subheader("📋 Category Overview")
        st.markdown("*Professional categories represented:*")
        
        total_participants = sum(item[1] for item in stats['category_breakdown'])
        for category, count in stats['category_breakdown'][:5]:
            percentage = (count / total_participants) * 100
            st.write(f"**{category}**: {count} people ({percentage:.1f}%)")
    
    st.divider()
    
    # Category analysis
    st.subheader("📊 Category Analysis")
    
    # Create category mapping for detailed analysis
    people_df['categories_list'] = people_df['categories'].fillna('').str.split(';')
    
    # Flatten categories for analysis
    all_categories = []
    for cats in people_df['categories_list']:
        if isinstance(cats, list):
            all_categories.extend([cat.strip() for cat in cats if cat.strip()])
    
    category_counter = Counter(all_categories)
    
    if category_counter:
        cat_df = pd.DataFrame(list(category_counter.items()), columns=['Category', 'Count'])
        cat_df = cat_df.sort_values('Count', ascending=False)
        
        fig_cat_detail = px.treemap(
            cat_df,
            path=['Category'],
            values='Count',
            title="Professional Categories Distribution (Treemap)",
            color='Count',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_cat_detail, use_container_width=True)

# Trends & Analysis Page
elif page == "📈 Trends & Analysis":
    st.header("📈 Trends & Deep Analysis")
    
    # Movie year distribution
    st.subheader("🎬 Movie Selection Trends by Year")
    
    # Convert year data to proper format for analysis
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    movies_df = movies_df.dropna(subset=['year'])
    
    # Year distribution
    year_dist = movies_df['year'].value_counts().sort_index()
    
    fig_year_trend = px.line(
        x=year_dist.index,
        y=year_dist.values,
        title="Movie Selections by Release Year",
        labels={'x': 'Release Year', 'y': 'Number of Selections'},
        markers=True
    )
    fig_year_trend.add_scatter(
        x=year_dist.index,
        y=year_dist.values,
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Data Points'
    )
    st.plotly_chart(fig_year_trend, use_container_width=True)
    
    st.divider()
    
    # Popular movies by decade
    st.subheader("🗓️ Popular Movies by Decade")
    
    movies_df['decade'] = (movies_df['year'] // 10) * 10
    decade_counts = movies_df['decade'].value_counts().sort_index()
    
    fig_decade = px.bar(
        x=decade_counts.index.astype(str) + 's',
        y=decade_counts.values,
        title="Movie Selections by Decade",
        labels={'x': 'Decade', 'y': 'Number of Selections'},
        color=decade_counts.values,
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_decade, use_container_width=True)

# Individual Ballots Page
elif page == "🔍 Individual Ballots":
    st.header("🔍 Individual Ballots Explorer")
    
    # Person selector and category filter in same row
    col1, col2 = st.columns([2, 1])
    
    with col2:
        category_filter = st.selectbox(
            "Filter by category:",
            ['All'] + [cat[0] for cat in stats['category_breakdown']]
        )
    
    with col1:
        # Filter person names based on category selection
        if category_filter != 'All':
            filtered_people = people_df[people_df['categories'].str.contains(category_filter, na=False)]
            person_names = sorted(filtered_people['name'].tolist())
        else:
            person_names = sorted(people_df['name'].tolist())
        
        selected_person = st.selectbox("Select a person to view their ballot:", person_names)
    
    if selected_person:
        # Get person info
        person_info = people_df[people_df['name'] == selected_person].iloc[0]
        person_movies = movies_df[movies_df['person_name'] == selected_person].sort_values('rank')
        
        # Display person info
        st.subheader(f"🎭 {person_info['name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Bio:** {person_info['bio']}")
            st.markdown(f"**Categories:** {person_info['categories']}")
            
            if person_info['top_choice'] == 'TRUE':
                st.success("⭐ This person has a highlighted top choice!")
            
            if pd.notna(person_info['notes']) and person_info['notes'].strip():
                st.info(f"**Notes:** {person_info['notes']}")
        
        with col2:
            st.metric("Movies Selected", len(person_movies))
            if len(person_movies) > 0:
                avg_year = person_movies['year'].mean()
                st.metric("Average Movie Year", f"{avg_year:.0f}")
        
        # Display movies
        if len(person_movies) > 0:
            st.subheader("🎬 Movie Selections")
            
            # Create a single row layout for all movies
            cols = st.columns(10)
            
            for i in range(len(person_movies)):
                if i < 10:  # Only show up to 10 movies
                    movie = person_movies.iloc[i]
                    image_url = get_movie_image_url(movie['nyt_id'])
                    
                    with cols[i]:
                        # Create a container for each movie
                        with st.container():
                            if image_url:
                                st.image(image_url, use_container_width=True)
                            else:
                                st.write("🎬")
                            
                            st.markdown(f"**#{movie['rank']} {movie['title']}**")
                            st.write(f"📅 {movie['year']}")
                            
                            if pd.notna(movie['review_url']):
                                st.markdown(f"[NYT Review]({movie['review_url']})")
                            
                            if pd.notna(movie['imdb_id']):
                                st.markdown(f"[IMDB](https://www.imdb.com/title/{movie['imdb_id']})")
        else:
            st.warning("No movies found for this person.")
    


# Interactive Explorer Page
elif page == "📊 Interactive Explorer":
    st.header("📊 Interactive Data Explorer")
    
    # Filters
    st.subheader("🔧 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Year range filter
        year_min = int(movies_df['year'].min())
        year_max = int(movies_df['year'].max())
        year_range = st.slider(
            "Select year range:",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1
        )
    
    with col2:
        # Category filter
        all_categories = ['All'] + [cat[0] for cat in stats['category_breakdown']]
        selected_categories = st.multiselect(
            "Select categories:",
            all_categories,
            default=['All']
        )
    
    with col3:
        # Minimum selections filter
        min_selections = st.slider(
            "Minimum selections per movie:",
            min_value=1,
            max_value=30,
            value=1,
            help="Only show movies selected by at least this many people"
        )
    
    # Apply filters
    filtered_movies = movies_df[
        (movies_df['year'] >= year_range[0]) & 
        (movies_df['year'] <= year_range[1])
    ]
    
    if 'All' not in selected_categories:
        # Filter by category
        filtered_people = people_df[
            people_df['categories'].str.contains('|'.join(selected_categories), na=False)
        ]
        filtered_movies = filtered_movies[
            filtered_movies['person_name'].isin(filtered_people['name'])
        ]
    
    # Count movie selections
    movie_counts = filtered_movies['title'].value_counts()
    movie_counts = movie_counts[movie_counts >= min_selections]
    
    st.divider()
    
    # Display results
    st.subheader("📊 Filtered Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Movies in filter", len(movie_counts))
        st.metric("Total selections", len(filtered_movies))
    
    with col2:
        if len(filtered_movies) > 0:
            avg_year = filtered_movies['year'].mean()
            st.metric("Average year", f"{avg_year:.0f}")
        
        unique_people = len(filtered_movies['person_name'].unique())
        st.metric("People represented", unique_people)
    
    # Top movies with images
    if len(movie_counts) > 0:
        st.subheader("🎬 Top Movies Gallery (Filtered)")
        
        # Show top 8 movies with images
        top_movies_display = movie_counts.head(8)
        cols = st.columns(4)
        
        for idx, (movie_title, count) in enumerate(top_movies_display.items()):
            col_index = idx % 4
            
            # Get movie details
            movie_info = filtered_movies[filtered_movies['title'] == movie_title].iloc[0]
            image_url = get_movie_image_url(movie_info['nyt_id'])
            
            with cols[col_index]:
                if image_url:
                    st.image(image_url, caption=f"{movie_title}", use_container_width=True)
                else:
                    st.write(f"🎬 **{movie_title}**")
                
                st.write(f"🗳️ **{count} selections**")
                st.write(f"📅 {movie_info['year']}")
        
        st.divider()
        
        # Bar chart
        top_movies = movie_counts.head(15)
        
        fig_filtered = px.bar(
            x=top_movies.values,
            y=top_movies.index,
            orientation='h',
            title=f"Top Movies (Filtered Results)",
            labels={'x': 'Number of Selections', 'y': 'Movie'},
            color=top_movies.values,
            color_continuous_scale='viridis'
        )
        fig_filtered.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_filtered, use_container_width=True)
        
        # Data table
        st.subheader("📋 Detailed Data")
        
        # Create summary table
        movie_details = []
        for movie, count in movie_counts.head(20).items():
            movie_data = filtered_movies[filtered_movies['title'] == movie]
            avg_rank = movie_data['rank'].mean()
            years = movie_data['year'].unique()
            
            movie_details.append({
                'Movie': movie,
                'Selections': count,
                'Average Rank': f"{avg_rank:.1f}",
                'Year': years[0] if len(years) == 1 else f"{years[0]} (varies)",
                'Selected By': ', '.join(movie_data['person_name'].head(5).tolist()) + 
                            ('...' if len(movie_data) > 5 else '')
            })
        
        details_df = pd.DataFrame(movie_details)
        st.dataframe(details_df, use_container_width=True)
    
    else:
        st.warning("No movies match the current filters. Try adjusting your selection.")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📊 Built with Streamlit | Data from NYT "Best Movies of the 21st Century" Survey</p>
        <p>🎬 Featuring ballots from 119 film industry professionals</p>
    </div>
    """, 
    unsafe_allow_html=True
) 