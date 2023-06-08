 # Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 


# Load EDA
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

def load_data(data):
	df = pd.read_csv(data)
	return df 


# Fxn
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat = count_vect.fit_transform(data)
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat



# Recommendation Sys
@st.cache_resource
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
	# indices of the ad
	ad_indices = pd.Series(df.index,index=df['text_column']).drop_duplicates()
	# Index of ad
	idx = ad_indices[title]

	# Look into the cosine matr for that index
	sim_scores =list(enumerate(cosine_sim_mat[idx]))
	sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
	selected_ad_indices = [i[0] for i in sim_scores[1:]]
	selected_ad_scores = [i[0] for i in sim_scores[1:]]

	# Get the dataframe & title
	result_df = df.iloc[selected_ad_indices]
	result_df['similarity_score'] = selected_ad_scores
	final_recommended_ads = result_df[['text_column','similarity_score','ad_creative_link_titles','ad_creative_link_descriptions', 'ad_creative_link_titles_2','ad_delivery_start_time','ad_delivery_stop_time','delivery_by_region','demographic_distribution','estimated_audience_size_lower_bound','estimated_audience_size_upper_bound','impressions_lower_bound',
				'impressions_upper_bound',
				'page_name',
				'publisher_platforms',
				'spend_lower_bound',
				'spend_upper_bound',
				'days_publicated',
				'polaridad',
				'sentimiento',
				'translated_column']]
	return final_recommended_ads.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;"></span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;"></span><a href="{}",target="_blank">Link_2</a></p>
<p style="color:blue;"><span style="color:black;"></span><a href="{}",target="_blank">Link_Descripción</a></p>
<p style="color:blue;"><span style="color:black;"></span><a href="{}",target="_blank">Link_Título_2</a></p>
<p style="color:blue;"><span style="color:black;">Fecha_Inicio:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Fecha_Fin:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Region:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Distribución_Demográfica:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Audiencia_Esperada_Menor:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Audiencia_Esperada_Mayor:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Impresiones_Esperadas_Menor:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Impresiones_Esperadas_Mayor:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Página:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Plataformas:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Gasto_Menor:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Gasto_Mayor:</span>{}</p> 
<p style="color:blue;"><span style="color:black;">Días_Publicación:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Polaridad:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Sentimiento:</span>{}</p>
<p style="color:blue;"><span style="color:black;"> Traducción:</span>{}</p>


</div>
"""

# Search For Ad 
@st.cache_resource
def search_term_if_not_found(term,df):
	result_df = df[df['text_column'].str.contains(term)]
	return result_df


def main():

	st.title("Recomendación")

	menu = ["Home","Recommend","GPT-3","About",]
	choice = st.sidebar.selectbox("Menu",menu)

	df = load_data("data/ADS.csv")

	if choice == "Home":
		st.subheader("Home")
		st.dataframe(df.head(10))


	elif choice == "Recommend":
		st.subheader("Recommend Ads")
		cosine_sim_mat = vectorize_text_to_cosine_mat(df['text_column'])
		search_term = st.text_input("Search")
		num_of_rec = st.sidebar.number_input("Number",4,30,7)
		if st.button("Recommend"):
			if search_term is not None:
				try:
					results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
					with st.expander("Results as JSON"):
						results_json = results.to_dict('index')
						st.write(results_json)

					for row in results.iterrows():
						rec_title = row[1][0]
						rec_score = row[1][1]
						rec_link = row[1][2]
						rec_link_2 = row[1][3]
						rec_link_description = row[1][4]
						rec_link_title_2 = row[1][5]
						rec_start_date = row[1][6]
						rec_stop_date = row[1][7]
						rec_region = row[1][8]
						rec_demographic = row[1][9]
						rec_lower_bound = row[1][10]
						rec_upper_bound = row[1][11]
						rec_impresions_lower_bound = row[1][12]
						rec_impresions_upper_bound = row[1][13]
						rec_page_name = row[1][14]
						rec_publisher_platforms = row[1][15]
						rec_spend_lower_bound = row[1][16]
						rec_spend_upper_bound = row[1][17]
						rec_days_publicated = row[1][18]
						rec_polaridad = row[1][19]
						rec_sentimiento = row[1][20]
						rec_translated_column = row[1][21]
						stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_link,rec_link_2,rec_link_description,rec_link_title_2,rec_start_date,rec_stop_date,rec_region,rec_demographic,rec_lower_bound,
									rec_upper_bound,
									rec_impresions_lower_bound,
									rec_impresions_upper_bound, 
									rec_page_name,
									rec_publisher_platforms,
									rec_spend_lower_bound,
									rec_spend_upper_bound,
									rec_days_publicated,
									rec_polaridad,
									rec_sentimiento,
									rec_translated_column

								  ),height=350)
				except:
					results= "Not Found"
					st.warning(results)
					st.info("Suggested Options include")
					result_df = search_term_if_not_found(search_term,df)
					st.dataframe(result_df)

	elif choice == "GPT-3":
     
		st.subheader("GPT-3")
		


				# How To Maximize Your Profits Options Trading

	else:
		st.subheader("About")
		st.text("Built with Streamlit & Pandas")


if __name__ == '__main__':
	main()