import { Routes } from '@angular/router';


export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: '/predict-genre',
  },
  {
    path: 'predict-genre',
    loadComponent: () =>
      import('./pages/predict-genre/predict-genre.component').then(
        (c) => c.PredictGenreComponent
      ),
  },
];
