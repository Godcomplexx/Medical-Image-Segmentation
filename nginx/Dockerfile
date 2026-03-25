FROM nginx:1.28.0-alpine

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx/nginx.conf /etc/nginx/nginx.conf

COPY nginx/conf.d/ /etc/nginx/conf.d/

COPY nginx/includes/ /etc/nginx/includes/

CMD ["nginx", "-g", "daemon off;"]