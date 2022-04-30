
/**
 * MovieService, return information of a specific movie
 */

public class MovieService extends HttpServlet {
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws IOException {
        try {
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            //get movie id via url parameter
            String movieId = request.getParameter("id");

            //get movie object from DataManager
            Movie movie = DataManager.getInstance().getMovieById(Integer.parseInt(movieId));

            //convert movie object to json format and return
            if (null != movie) {
                ObjectMapper mapper = new ObjectMapper();
                String jsonMovie = mapper.writeValueAsString(movie);
                response.getWriter().println(jsonMovie);
            }else {
                response.getWriter().println("");
            }

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }
}
