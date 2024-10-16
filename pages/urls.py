from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/<int:file_id>/', views.upload, name='view_csv_content'),
    path('delete-file/<int:file_id>/', views.delete_file, name='delete_file'),
    path('dash/', views.list_files, name='dash'),
    path('dash/', views.dash, name='dash'),
    path('pretraitement/', views.pretraitement_dataset, name='pretraitement'),
    # path('pretraitement/', views.pretraitement, name='pretraitement'),
    path('algo/', views.algo, name='algo'),
    path('perforamnces/', views.perforamnces, name='perforamnces'),
    path('parametres/', views.parametres, name='parametres'),
    path('get_column_details/', views.get_column_details, name='get_column_details'),
    path('train_model/', views.train_model, name='train_model'),
    path('predict/', views.predict_view, name='predict_view'),
    path('predict_resul/', views.predict, name='predict_resul'),


]
