def recomendar_por_idx(modelo, X, indice_libro, n_recomendaciones = 10):
    distancias, indices = modelo.kneighbors(X[indice_libro].reshape(1, -1), n_neighbors = n_recomendaciones + 1)
    return indices[0][1:], distancias[0][1:]

def recomendar_por_libro(modelo, libros_df, X, titulo_autor, titulo_a_indice, cant_recomendaciones = 5):

    #Primer caso: No encontramos ningun libro con esos datos
    if (titulo_autor not in titulo_a_indice):
        print("No se encontro ningun libro con esos datos.")
        return None
    
    #Segundo caso: se encontro el libro
    indice_libro = titulo_a_indice[titulo_autor]
    
    #Obtengo las recomendaciones:
    libros_recomendados, distancias = recomendar_por_idx(modelo, X, indice_libro, cant_recomendaciones)

    recomendaciones = libros_df.iloc[libros_recomendados].copy()
    recomendaciones["distancia"] = distancias
    recomendaciones["similitud"] = (1 - recomendaciones["distancia"]) * 100 #Usamos esta formula porque estamos usando la metrica "cosine"
    columnas = ["title", "author_name", "publication_year", "average_rating", "similitud"]
    return recomendaciones[columnas]

def mostrar_recomendaciones(recomendaciones):
    print(">>> Estoy mostrando recomendaciones")
    html = """
    <h2 style="
        color:#2E86C1;
        font-family:Arial;
        margin-bottom:20px;
    ">
        💡 Libros recomendados
    </h2>

    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:20px;
        justify-content:flex-start;
        align-items:flex-start;
    ">
    """

    for _, libro in recomendaciones.iterrows():

        html += f"""
        <div style="
            width:270px;
            min-height:280px;
            background:#F8F9FA;
            border:1px solid #D6DBDF;
            border-radius:15px;
            padding:18px;
            box-shadow:3px 3px 12px rgba(0,0,0,0.12);
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            transition:0.2s;
        ">

            <div>

                <h3 style="
                    margin-top:0;
                    height:60px;
                    overflow:hidden;
                    color:#2C3E50;
                    font-family:Arial;
                ">
                    📖 {libro["title"]}
                </h3>

                <hr style="border:0;border-top:1px solid #E5E7E9;">

                <p style="
                    font-size:15px;
                    line-height:1.6;
                    color:#34495E;
                    font-family:Arial;
                ">

                👤 <b>Autor</b><br>
                {libro["author_name"]}<br><br>

                📅 <b>Año</b><br>
                {int(libro["publication_year"])}

                </p>

            </div>

            <div>

                <span style="
                    background:#FFF3CD;
                    color:#856404;
                    padding:6px 10px;
                    border-radius:8px;
                    font-weight:bold;
                    font-size:14px;
                ">
                    ⭐ {libro["average_rating"]:.2f}
                </span>

                <span style="
                    background:#D4EDDA;
                    color:#155724;
                    padding:6px 10px;
                    border-radius:8px;
                    font-weight:bold;
                    font-size:14px;
                    margin-left:8px;
                ">
                    🎯 {libro["similitud"]:.1f}%
                </span>

            </div>

        </div>
        """

    html += "</div>"

    #display(HTML(html))
    return html

