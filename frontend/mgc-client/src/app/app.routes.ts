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
  {
    path: 'result-view',
    loadComponent: () =>
      import('./pages/result-view/result-view.component').then(
        (c) => c.ResultViewComponent
      ),
  },
];
